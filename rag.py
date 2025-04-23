import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, Document
from transformers.cache_utils import DynamicCache
import cag.dataset as cagds
import cag.similarity as cagsim
import argparse
import os
from transformers import BitsAndBytesConfig
import logging
from bert_score import score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


"""Hugging Face Llama model"""

global model_name, model, tokenizer
global rand_seed

# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


from time import time
from llama_index.core import Settings

def getOpenAIRetriever(documents: list[Document], similarity_top_k: int = 1):
    """OpenAI RAG model"""
    import openai
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found")
    openai.api_key = OPENAI_API_KEY
    # from llama_index.llms.openai import OpenAI
    # Settings.llm = OpenAI(model="gpt-3.5-turbo")
    
    from llama_index.embeddings.openai import OpenAIEmbedding
    # Set the embed_model in llama_index
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=OPENAI_API_KEY, title="openai-embedding")
    # model_name: "text-embedding-3-small", "text-embedding-3-large"
    
    # Create the OpenAI retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    OpenAI_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    logger.info(f"OpenAI retriever prepared in {t2 - t1:.2f} seconds.")
    return OpenAI_retriever, t2 - t1
    

def getGeminiRetriever(documents: list[Document], similarity_top_k: int = 1):
    """Gemini Embedding RAG model"""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found")
    from llama_index.embeddings.gemini import GeminiEmbedding
    model_name = "models/embedding-001"
    # Set the embed_model in llama_index
    Settings.embed_model = GeminiEmbedding( model_name=model_name, api_key=GOOGLE_API_KEY, title="gemini-embedding")
    
    # Create the Gemini retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    Gemini_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    logger.info(f"Gemini retriever prepared in {t2 - t1:.2f} seconds.")
    return Gemini_retriever, t2 - t1
    
def getBM25Retriever(documents: list[Document], similarity_top_k: int = 1):
    from llama_index.core.node_parser import SentenceSplitter  
    from llama_index.core.retrievers import BM25Retriever
    import Stemmer

    splitter = SentenceSplitter(chunk_size=512)
    
    t1 = time()
    nodes = splitter.get_nodes_from_documents(documents)
    # We can pass in the index, docstore, or list of nodes to create the retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    t2 = time()
    bm25_retriever.persist("./bm25_retriever")

    return bm25_retriever, t2 - t1

def getJinaRetriever(documents: list[Document], similarity_top_k: int = 1):
    """Jina Embedding model"""
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY not found")
    try:
        from llama_index.embeddings.jinaai import JinaEmbedding
        model_name = "jina-embeddings-v3"
        Settings.embed_model = JinaEmbedding(
            api_key=JINA_API_KEY,
            model=model_name,
            task="retrieval.passage",
        )

        # Create the Jina retriever
        t1 = time()
        index = VectorStoreIndex.from_documents(documents)
        Jina_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        t2 = time()
        logger.info(f"Jina retriever prepared in {t2 - t1:.2f} seconds.")
        return Jina_retriever, t2 - t1
    except ImportError:
        logger.error("Failed to import JinaEmbedding. Please install jinaai package.")
        raise
    except Exception as e:
        logger.error(f"Error creating Jina retriever: {str(e)}")
        raise

    
def calculate_confidence_score(model, inputs, response):
    """Calculate a confidence score based on model outputs."""
    with torch.no_grad():
        # Tokenize inputs
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(model.device)
        
        # Get model outputs
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        
        # Base confidence from probability distribution
        base_confidence = probs.max().item()
        
        # Calculate entropy of the distribution
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        entropy_penalty = min(entropy / 5.0, 1.0)  # Normalize entropy
        
        # Combine factors
        confidence = base_confidence * (1 - entropy_penalty)
        return min(max(confidence, 0.0), 1.0)

def detect_hallucination(response, context, question):
    """Detect potential hallucination in the response."""
    # 1. Semantic similarity with context
    context_similarity = score([response], [context], lang='en', verbose=False)[2].mean().item()
    
    # 2. Keyword overlap
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    keyword_overlap = len(response_words.intersection(context_words)) / len(response_words) if response_words else 0
    
    # 3. Question relevance
    question_words = set(question.lower().split())
    response_relevance = len(response_words.intersection(question_words)) / len(question_words) if question_words else 0
    
    # 4. Factual consistency check
    factual_consistency = 1.0
    if "cannot answer" in response.lower() or "not enough information" in response.lower():
        factual_consistency = 0.0
    
    # Combine metrics with weights
    hallucination_score = (
        0.4 * (1 - context_similarity) +
        0.3 * (1 - keyword_overlap) +
        0.2 * (1 - response_relevance) +
        0.1 * (1 - factual_consistency)
    )
    
    return min(max(hallucination_score, 0.0), 1.0)

def rag_test(args):
    """Run RAG test"""
    global model, tokenizer
    
    # Load dataset
    text_list, dataset = cagds.get(
        args.dataset,
        max_knowledge=args.maxKnowledge,
        max_questions=args.maxQuestion
    )
    
    if not text_list or not dataset:
        print("Failed to load dataset")
        return
        
    # Create documents for indexing
    documents = [Document(text=text) for text in text_list]
    
    # Get retriever based on index type
    if args.index == "openai":
        retriever, _ = getOpenAIRetriever(documents, args.topk)
    elif args.index == "faiss":
        retriever, _ = getFaissRetriever(documents, args.topk)
    elif args.index == "bm25":
        retriever, _ = getBM25Retriever(documents, args.topk)
    else:
        print(f"Unknown index type: {args.index}")
        return
        
    # Process questions
    results = []
    questions = []
    answers = []
    for q, a in dataset:
        questions.append(q)
        answers.append(a)
    total_questions = len(questions)
    
    for i, (question, ground_truth) in enumerate(zip(questions, answers)):
        print(f"\nProcessing question {i+1}/{total_questions}")
        
        # Get relevant documents
        retrieved_nodes = retriever.retrieve(question)
        if not retrieved_nodes:
            print("No relevant documents found")
            continue
            
        # Prepare context from retrieved documents
        context = "\n".join([node.text for node in retrieved_nodes])
        
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
            # Generate response
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Calculate confidence and hallucination scores
            confidence = calculate_confidence_score(model, prompt, response)
            hallucination_score = detect_hallucination(response, context, question)
            
            # Adjust response based on scores
            if confidence < 0.6 or hallucination_score > 0.4:
                response = "I cannot provide a confident answer based on the available information."
            elif confidence < 0.8:
                response = f"I'm somewhat confident, but not entirely sure: {response}"
                
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
            'context': context
        })
        
    # Write results to file
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            for result in results:
                f.write(f"Question: {result['question']}\n")
                f.write(f"Ground Truth: {result['ground_truth']}\n")
                f.write(f"Response: {result['response']}\n")
                f.write(f"Confidence: {result['confidence']:.2f}\n")
                f.write(f"Hallucination Score: {result['hallucination_score']:.2f}\n")
                f.write("\nContext:\n")
                f.write(f"{result['context']}\n")
                f.write("\n" + "="*80 + "\n\n")
                
    print(f"\nProcessed {len(results)} questions")
    print(f"Results written to {args.output}")


# Define quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True  # Use nested quantization
)

def load_quantized_model(model_name, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically choose best device
        trust_remote_code=True,     # Required for some models
        token=hf_token
    )
    
    return tokenizer, model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="openai", choices=["openai", "faiss", "bm25"])
    parser.add_argument("--dataset", type=str, default="squad-dev", choices=["trivia-qa-train", "natural-questions-train", "squad-dev", "hotpotqa"])
    parser.add_argument("--similarity", type=str, default="bertscore", choices=["bertscore", "cosine"])
    parser.add_argument("--maxKnowledge", type=int, default=32)
    parser.add_argument("--maxParagraph", type=int, default=32)
    parser.add_argument("--maxQuestion", type=int, default=100)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--modelname", type=str, default="facebook/opt-125m")
    parser.add_argument("--output", type=str, default="./results/squad/test_result_rag.txt")
    parser.add_argument("--randomSeed", type=int, default=None)
    parser.add_argument("--quantized", action="store_true", help="Whether to use quantized model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    print("maxKnowledge", args.maxKnowledge, "maxParagraph", args.maxParagraph, "maxQuestion", args.maxQuestion, "randomSeed", args.randomSeed)
    
    model_name = args.modelname
    rand_seed = args.randomSeed if args.randomSeed != None else None
    
    if args.quantized:
        tokenizer, model = load_quantized_model(model_name=model_name, hf_token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN
        )
    
    def unique_path(path, i=0):
        if os.path.exists(path):
            return unique_path(path + "_" + str(i), i + 1)
        return path
    
    if os.path.exists(args.output):
        args.output = unique_path(args.output)
        
    rag_test(args)
