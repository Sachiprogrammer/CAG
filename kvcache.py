import torch
import argparse
import os
from cag.dataset import get
import cag.dataset as cagds
import cag.similarity as cagsim
from time import time
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import logging 
from typing import Optional, Union, List, Tuple, Dict, Any
import bitsandbytes as bnb
import json
import gc
import re
from bert_score import score
import torch.nn as nn
import math
import faiss
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# Set memory management for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit for memory allocations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Limit memory splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")


"""Hugging Face Llama model"""

global model_name, model, tokenizer
global rand_seed


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Configure quantization for MPS
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for MPS
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False
)

# Add memory management
def clear_gpu_memory():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    import gc
    gc.collect()


"""KV Cache test"""
# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


def generate(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    max_new_tokens: int = 300
) -> torch.Tensor:
    """
    Generate text with MPS memory-efficient decoding.
    """
    origin_ids = input_ids
    output_ids = input_ids.clone()
    next_token = input_ids.to(model.device)
    
    # Process in smaller chunks
    chunk_size = 50  # Process fewer tokens at a time

    with torch.no_grad():
        for _ in range(0, max_new_tokens, chunk_size):
            # Clear memory before processing chunk
            clear_gpu_memory()
            
            outputs = model(
                input_ids=next_token, 
                past_key_values=past_key_values,
                use_cache=True,
                temperature=0.7,  # Add temperature for more natural responses
                top_p=0.9,  # Add nucleus sampling
                top_k=50,  # Add top-k sampling
                do_sample=True  # Enable sampling
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature and sampling
            next_token_logits = next_token_logits / 0.7  # Apply temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Top-k sampling
            top_k_values, top_k_indices = torch.topk(next_token_probs, k=50, dim=-1)
            next_token_probs = torch.zeros_like(next_token_probs).scatter_(-1, top_k_indices, top_k_values)
            
            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum_probs > 0.9
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0
            sorted_probs[mask] = 0.0
            next_token_probs = torch.zeros_like(next_token_probs).scatter_(-1, sorted_indices, sorted_probs)
            
            # Sample from the distribution
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            past_key_values = outputs.past_key_values
            output_ids = torch.cat([output_ids, next_token], dim=1)

            # Check for end of sequence
            if isinstance(model.config.eos_token_id, list):
            if next_token.item() in model.config.eos_token_id:
                break
            else:
                if next_token.item() == model.config.eos_token_id:
                    break
            
            clear_gpu_memory()
    
    return output_ids[:, origin_ids.shape[-1]:]


def preprocess_knowledge(
    model,
    tokenizer,
    prompt: str,
    chunk_size: int = 256  # Reduced chunk size
) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG with MPS memory management.
    """
    clear_gpu_memory()
    
    # Split the prompt into smaller chunks
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    num_chunks = (tokens.shape[1] + chunk_size - 1) // chunk_size
    past_key_values = None
    
    print(f"Processing {tokens.shape[1]} tokens in {num_chunks} chunks...")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, tokens.shape[1])
        chunk = tokens[:, start_idx:end_idx].to(model.device)
        
        # Clear cache before processing
        clear_gpu_memory()
        
    with torch.no_grad():
        outputs = model(
                input_ids=chunk,
            past_key_values=past_key_values,
                use_cache=True
            )
            
            if past_key_values is None:
                past_key_values = DynamicCache()
                for layer_idx in range(len(outputs.past_key_values)):
                    past_key_values.key_cache.append(outputs.past_key_values[layer_idx][0])
                    past_key_values.value_cache.append(outputs.past_key_values[layer_idx][1])
            else:
                for layer_idx in range(len(outputs.past_key_values)):
                    past_key_values.key_cache[layer_idx] = torch.cat([
                        past_key_values.key_cache[layer_idx],
                        outputs.past_key_values[layer_idx][0]
                    ], dim=-2)
                    past_key_values.value_cache[layer_idx] = torch.cat([
                        past_key_values.value_cache[layer_idx],
                        outputs.past_key_values[layer_idx][1]
                    ], dim=-2)
        
        print(f"Processed chunk {i+1}/{num_chunks}")
        clear_gpu_memory()
    
    return past_key_values


def write_kv_cache(kv: DynamicCache, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    """
    Write the KV Cache to a file.
    """
    torch.save(kv, path)


def clean_up(kv: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length.
    """
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]


def read_kv_cache(path: str) -> Optional[DynamicCache]:
    """
    Read the KV Cache from a file. If the cache file is invalid or empty, return None.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        kv = torch.load(path, weights_only=True)
        return kv
    else:
        # Regenerate cache if it doesn't exist or is too small
        return None


def prepare_kvcache(documents, filepath: str = "./data_cache/cache_knowledges.pt", answer_instruction: Optional[str] = None):
    # Prepare the knowledges kvcache
    global model, tokenizer, model_name

    if answer_instruction is None:
        answer_instruction = """Answer the question based ONLY on the provided context. If the context doesn't contain enough information to answer the question, respond with "I cannot answer based on the available information." Do not make assumptions or include information not present in the context."""
    
    # Enhanced context preparation
    knowledges = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant that provides accurate answers based strictly on the given context.
    Follow these rules:
    1. Only use information explicitly stated in the context
    2. If unsure or if information is missing, say "I cannot answer based on the available information"
    3. Do not make assumptions or include external knowledge
    4. Be concise and direct in your answers
    5. If multiple interpretations are possible, choose the most relevant one based on the context
    <|eot_id|>
    {documents}
    """

    # Check if cache exists and is valid
    kv = read_kv_cache(filepath)
    if kv is not None:
        # Validate cache contents
        try:
            # Test cache with a simple query
            test_input = tokenizer("test", return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(input_ids=test_input.input_ids, past_key_values=kv)
            return kv, 0
        except Exception as e:
            print(f"Cache validation failed: {str(e)}. Regenerating cache...")
            kv = None

    # Initialize model and tokenizer if not already set
    if model is None or tokenizer is None:
        print(f"Loading model {model_name} with MPS optimizations...")
        model, tokenizer = load_model(model_name, device)

    t1 = time()
    kv = preprocess_knowledge(model, tokenizer, knowledges)
    t2 = time()
    write_kv_cache(kv, filepath)
    return kv, t2 - t1


def load_model(model_name, device):
    """Load model with MPS optimizations and memory management."""
    print(f"Loading model {model_name} with MPS optimizations...")
    
    # Configure model loading for MPS
    model_kwargs = {
        "torch_dtype": torch.float32,  # Use float32 for better compatibility
        "device_map": "mps",
        "max_memory": {"mps": "4GB"},
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    # Load tokenizer with padding configuration
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Convert model to float32 for better compatibility
    model = model.to(dtype=torch.float32)
    
    # Enable memory efficient settings
    model.config.use_cache = True
    
    return model, tokenizer


def clean_text(text):
    """
    Clean text by removing HTML tags, special characters, and extra whitespace.
    Also handles special formatting cases.
    """
    # Convert to string if not already
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown-style formatting
    text = re.sub(r'[`*_]', '', text)
    
    # Replace quotes with standard quotes
    text = text.replace('``', '"').replace("''", '"')
    
    # Remove parentheses around years
    text = re.sub(r'\((\d{4})\)', r'\1', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove leading/trailing punctuation
    text = text.strip('.,!? -')
    
    return text.strip()


def calculate_confidence_score(model, inputs, response):
    """Calculate a more sophisticated confidence score based on multiple factors."""
    with torch.no_grad():
        # Tokenize inputs
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(model.device)
        
        # Get model outputs
        outputs = model(input_ids=input_ids, past_key_values=None)
        logits = outputs.logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        
        # Base confidence from probability distribution
        base_confidence = probs.max().item()
        
        # Calculate entropy of the distribution
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        entropy_penalty = min(entropy / 5.0, 1.0)  # Normalize entropy
        
        # Calculate attention variance
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention = outputs.attentions[-1]  # Last layer attention
            attention_variance = torch.var(attention).item()
            attention_penalty = min(attention_variance / 0.5, 1.0)  # Normalize variance
        else:
            attention_penalty = 0.5
        
        # Combine factors
        confidence = base_confidence * (1 - entropy_penalty) * (1 - attention_penalty)
        return min(max(confidence, 0.0), 1.0)


def detect_hallucination(response, context, question):
    """Enhanced hallucination detection using multiple metrics."""
    # 1. Semantic similarity with context
    context_similarity = score([response], [context], lang='en', verbose=False)[2].mean().item()
    
    # 2. Keyword overlap
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    keyword_overlap = len(response_words.intersection(context_words)) / len(response_words) if response_words else 0
    
    # 3. Named entity consistency
    response_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response))
    context_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context))
    entity_consistency = len(response_entities.intersection(context_entities)) / len(response_entities) if response_entities else 0
    
    # 4. Question relevance
    question_words = set(question.lower().split())
    response_relevance = len(response_words.intersection(question_words)) / len(question_words) if question_words else 0
    
    # 5. Factual consistency check
    factual_consistency = 1.0
    if "cannot answer" in response.lower() or "not enough information" in response.lower():
        factual_consistency = 0.0
    
    # Combine metrics with weights
    hallucination_score = (
        0.3 * (1 - context_similarity) +
        0.2 * (1 - keyword_overlap) +
        0.2 * (1 - entity_consistency) +
        0.2 * (1 - response_relevance) +
        0.1 * (1 - factual_consistency)
    )
    
    return min(max(hallucination_score, 0.0), 1.0)


def kvcache_test(args):
    """Run KV cache test with enhanced knowledge caching and confidence scoring."""
    global model, tokenizer
    
    # Initialize knowledge cache
    knowledge_cache = KnowledgeCache(model_name=args.modelname)
    knowledge_cache.initialize()
    
    # Load dataset
    dataset = get(args.dataset, max_knowledge=args.maxKnowledge, max_questions=args.maxQuestion)
    if not dataset:
        print("Failed to load dataset")
        return
        
    documents, qa_pairs = dataset
    
    # Prepare knowledge cache
    print(f"Preparing knowledge cache with {len(documents)} documents...")
    for doc in documents:
        knowledge_cache.add_to_cache(doc[:100], doc)  # Use first 100 chars as key
        
    # Process questions
    results = []
    questions = []
    answers = []
    for q, a in qa_pairs:
        questions.append(q)
        answers.append(a)
    total_questions = len(questions)
    
    for i, (question, ground_truth) in enumerate(zip(questions, answers)):
        print(f"\nProcessing question {i+1}/{total_questions}")
        
        # Search for relevant knowledge
        relevant_knowledge = knowledge_cache.search(question, k=args.topk)
        if not relevant_knowledge:
            print("No relevant knowledge found")
            continue
            
        # Prepare context from relevant knowledge
        context = "\n".join([k['value'] for k in relevant_knowledge])
        
        # Generate response with enhanced prompt
        prompt = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant that provides accurate answers based on the given context. Follow these rules:
1. Read the context carefully
2. Find the most relevant information to answer the question
3. If the answer is in the context, provide it clearly and concisely
4. If you're not sure or the answer is not in the context, say "I cannot answer based on the available information"
5. Do not make up information or include external knowledge
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the context above.
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
        
        try:
            # Clear GPU memory before generation
            clear_gpu_memory()
            
            # Tokenize prompt with truncation
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512  # Limit input length
            ).input_ids.to(model.device)
            
            # Generate response with smaller max tokens
            output_ids = generate(
                model=model,
                input_ids=input_ids,
                past_key_values=None,
                max_new_tokens=50  # Limit output length
            )
            
            # Decode response
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Calculate confidence score
            confidence = calculate_confidence_score(
                model=model,
                inputs=prompt,
                response=response
            )
            
            # Detect hallucination
            hallucination_score = detect_hallucination(
                response=response,
                context=context,
                question=question
            )
            
            # Adjust response based on confidence and hallucination
            if confidence < 0.6 or hallucination_score > 0.4:
                response = "I cannot provide a confident answer based on the available information."
            elif confidence < 0.8:
                response = f"I'm somewhat confident, but not entirely sure: {response}"
                
            # Clear GPU memory after generation
            clear_gpu_memory()
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            response = "Error generating response"
            confidence = 0.0
            hallucination_score = 1.0
            
        # Store results
        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'response': response,
            'confidence': confidence,
            'hallucination_score': hallucination_score,
            'relevant_knowledge': relevant_knowledge
        })
        
    # Write results to file
    if args.output:
        with open(args.output, 'w') as f:
            for result in results:
                f.write(f"Question: {result['question']}\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Response: {result['response']}\n")
                f.write(f"Confidence: {result['confidence']:.2f}\n")
                f.write(f"Hallucination Score: {result['hallucination_score']:.2f}\n")
                f.write("\nRelevant Knowledge:\n")
                for k in result['relevant_knowledge']:
                    f.write(f"- Score: {k['score']:.2f}\n")
                    f.write(f"  {k['value'][:200]}...\n")
                f.write("\n" + "="*80 + "\n\n")
                
    print(f"\nProcessed {len(results)} questions")
    print(f"Results written to {args.output}")


def main():
    """Main function to run KV cache testing."""
    global model, tokenizer
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run KV cache testing')
    parser.add_argument('--kvcache', type=str, required=True, choices=['file', 'memory'],
                      help='KV cache storage type')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['natural-questions-train', 'natural-questions-test', 'natural-questions-eval',
                              'trivia-qa-train', 'trivia-qa-test', 'trivia-qa-eval'],
                      help='Dataset to use')
    parser.add_argument('--similarity', type=str, required=True,
                      choices=['bertscore', 'cosine', 'euclidean'],
                      help='Similarity metric to use')
    parser.add_argument('--maxKnowledge', type=int, default=32,
                      help='Maximum number of knowledge items to use')
    parser.add_argument('--maxQuestion', type=int, default=100,
                      help='Maximum number of questions to process')
    parser.add_argument('--modelname', type=str, required=True,
                      help='Name of the model to use')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    parser.add_argument('--usePrompt', action='store_true',
                      help='Whether to use prompt engineering')
    parser.add_argument('--topk', type=int, default=5,
                      help='Number of top knowledge items to retrieve')
    args = parser.parse_args()

    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model {args.modelname}...")
    model, tokenizer = load_model(args.modelname, device)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run test
    kvcache_test(args)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def attention(self, q, k, v, mask=None):
        # Ensure consistent dtype
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v), attn
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations with consistent dtype
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attn = self.attention(q, k, v, mask)
        
        # Concatenate heads and put through final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(x)


class KnowledgeCache:
    def __init__(self, model_name="facebook/opt-350m", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.cache = {}
        self.embeddings = {}
        self.index = None
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def initialize(self):
        """Initialize the model, tokenizer, and FAISS index."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        
        # Initialize FAISS index with improved parameters
        embedding_dim = 384  # MiniLM-L6-v2 embedding dimension
        self.index = faiss.IndexFlatL2(embedding_dim)
        
    def get_embedding(self, text):
        """Get embedding for a text with improved normalization and pooling."""
        if not text or not isinstance(text, str):
            return None
            
        try:
            # Use sentence transformer for embeddings
            embeddings = self.embedding_model.encode([text], convert_to_tensor=True, show_progress_bar=False)
            # Convert to numpy for FAISS
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return None
            
    def add_to_cache(self, key, value):
        """Add item to cache with improved indexing and deduplication."""
        if not key or not value:
            return False
            
        # Clean and normalize the key
        key = clean_text(str(key))
        value = clean_text(str(value))
        
        # Check for duplicates
        if key in self.cache:
            # If duplicate exists, skip it
            return True
            
        # Get embedding for new item
        embedding = self.get_embedding(value)
        if embedding is None:
            return False
            
        # Add to cache and index
        self.cache[key] = value
        self.embeddings[key] = embedding
        self.index.add(embedding)
        return True
        
    def search(self, query, k=5):
        """Enhanced search with improved ranking and filtering."""
        if not query or not isinstance(query, str):
            return []
            
        # Clean and normalize query
        query = clean_text(str(query))
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
            
        # Search in index
        distances, indices = self.index.search(query_embedding, k * 2)  # Get more results for filtering
        
        # Get keys from cache
        keys = list(self.cache.keys())
        results = []
        seen_values = set()  # Track seen values to avoid duplicates
        
        # Filter and rank results
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(keys):
                key = keys[idx]
                value = self.cache[key]
                
                # Skip if we've seen this value before
                if value in seen_values:
                    continue
                seen_values.add(value)
                
                # Calculate additional similarity metrics
                text_similarity = score([query], [value], lang='en', verbose=False)[2].mean().item()
                
                # Combine metrics for final score
                final_score = (1 - dist/max(distances[0])) * 0.6 + text_similarity * 0.4
                
                results.append({
                    'key': key,
                    'value': value,
                    'score': final_score,
                    'distance': float(dist),
                    'text_similarity': float(text_similarity)
                })
        
        # Sort by final score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
        
    def clear(self):
        """Clear the cache and reset the index."""
        self.cache.clear()
        self.embeddings.clear()
        if self.index is not None:
            self.index.reset()
            
    def __len__(self):
        return len(self.cache)


if __name__ == "__main__":
    main()
