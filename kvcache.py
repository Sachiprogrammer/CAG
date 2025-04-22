import torch
import argparse
import os
import cag.dataset as cagds
import cag.similarity as cagsim
from time import time
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import logging 
from typing import Optional, Union, List, Tuple, Dict, Any


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


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Configure quantization for GPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,  # Disable 4-bit quantization for MPS compatibility
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False
)

# Add memory management
def clear_gpu_memory():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping
        input_ids: Input token ids
        past_key_values: KV Cache for knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    embed_device = model.model.embed_tokens.weight.device

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token, 
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(embed_device)

            past_key_values = outputs.past_key_values

            output_ids = torch.cat([output_ids, next_token], dim=1)

            # Fix: Check if next_token is in eos_token_id list
            if isinstance(model.config.eos_token_id, list):
                if next_token.item() in model.config.eos_token_id:
                    break
            else:
                if next_token.item() == model.config.eos_token_id:
                    break
    return output_ids[:, origin_ids.shape[-1]:]


def preprocess_knowledge(
    model,
    tokenizer,
    prompt: str,
) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """
    clear_gpu_memory()  # Clear memory before processing
    embed_device = model.model.embed_tokens.weight.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    past_key_values = DynamicCache()
    
    # Process in smaller chunks if needed
    max_length = 512  # Limit sequence length
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )
    clear_gpu_memory()  # Clear memory after processing
    return outputs.past_key_values


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

    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    knowledges = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """
    # Get the knowledge cache
    t1 = time()
    kv = preprocess_knowledge(model, tokenizer, knowledges)
    print("kvlen: ", kv.key_cache[0].shape[-2])
    write_kv_cache(kv, filepath)
    t2 = time()
    logger.info(f"KV cache prepared in {t2 - t1:.2f} seconds.")
    return kv, t2 - t1


def kvcache_test(args: argparse.Namespace):
    # Reduce workload
    args.maxKnowledge = min(args.maxKnowledge, 3)  # Limit knowledge to 3
    args.maxParagraph = min(args.maxParagraph, 50)  # Limit paragraphs to 50
    args.maxQuestion = min(args.maxQuestion, 20)  # Limit questions to 20
    
    answer_instruction = "Answer the question with a super short answer."
    text_list, dataset = cagds.get(args.dataset, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)

    kvcache_path = "./data_cache/cache_knowledges.pt"
    
    # Clear memory before starting
    clear_gpu_memory()

    knowledges = '\n\n\n\n\n\n'.join(text_list)
    knowledge_cache, prepare_time = prepare_kvcache(knowledges, filepath=kvcache_path, answer_instruction=answer_instruction)
    kv_len = knowledge_cache.key_cache[0].shape[-2]
    print(f"KVcache prepared in {prepare_time} seconds")
    
    # Clear memory after cache preparation
    clear_gpu_memory()
    
    with open(args.output, "a") as f:
        f.write(f"KVcache prepared in {prepare_time} seconds\n")

    results = {
        "cache_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }

    dataset = list(dataset)  # Convert the dataset to a list
    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion is not None else len(dataset)
    
    # Process questions in smaller batches
    batch_size = 5
    for batch_start in range(0, max_questions, batch_size):
        batch_end = min(batch_start + batch_size, max_questions)
        batch = dataset[batch_start:batch_end]
        
        for id, (question, ground_truth) in enumerate(batch):
            # Clear memory before each question
            clear_gpu_memory()
            
            # Read the knowledge cache from the cache file
            cache_t1 = time()
            # if args.kvcache == "file":
            #     knowledge_cache = read_kv_cache(kvcache_path)

            # Not a good idea to use this method, as it will consume a lot of memory
            # if args.kvcache == "variable":
            #     knowledge_cache = documents_cache
            cache_t2 = time()

            # Generate Response for the question
            knowledges = '\n\n\n'.join(text_list)

            if args.usePrompt:
                prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {knowledges}
    ------------------------------------------------
    {answer_instruction}
    Question:
    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
                generate_t1 = time()
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                output = generate(model, input_ids, DynamicCache()) 
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
                generate_t2 = time()
            else:
                prompt = f"""
    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
                generate_t1 = time()
                clean_up(knowledge_cache, kv_len)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                output = generate(model, input_ids, knowledge_cache)
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
                generate_t2 = time()

            # print("D: ", knowledges)
            print("Q: ", question)
            print("A: ", generated_text)
            print("Ground Truth: ", ground_truth)
            print("-" * 80)
            
            # Calculate similarity
            similarity = cagsim.bert(generated_text, ground_truth)
            
            # Update results
            results["prompts"].append(question)
            results["responses"].append(generated_text)
            results["cache_time"].append(cache_t2 - cache_t1)
            results["generate_time"].append(generate_t2 - generate_t1)
            results["similarity"].append(similarity)
            
            with open(args.output, "a") as f:
                f.write(f"\nQuestion {id}: {question}\n")
                f.write(f"Answer: {generated_text}\n")
                f.write(f"Ground Truth: {ground_truth}\n")
                f.write("-" * 80 + "\n")
                f.write(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t cache time: {cache_t2 - cache_t1},\t generate time: {generate_t2 - generate_t1}\n")
                if len(results["similarity"]) > 0:
                    f.write(f"[{id}]: [Cumulative]: Semantic Similarity: {round(sum(results['similarity']) / len(results['similarity']), 5)},\t cache time: {sum(results['cache_time']) / len(results['cache_time'])},\t generate time: {sum(results['generate_time']) / len(results['generate_time'])}\n")

    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    print()
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"cache time: {avg_cache_time},\t generate time: {avg_generate_time}")
    print()
    with open(args.output, "a") as f:
        f.write("\n")
        f.write(f"Result for {args.output}\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"cache time: {avg_cache_time},\t generate time: {avg_generate_time}\n")


def load_model_with_memory_optimization(model_name, hf_token):
    clear_gpu_memory()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        low_cpu_mem_usage=True,
        max_memory={0: "4GB"},  # Limit memory usage
        offload_folder="offload",  # Enable disk offloading
        offload_state_dict=True  # Enable state dict offloading
    )
    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxKnowledge", type=int, default=1)  # Further reduced
    parser.add_argument("--maxParagraph", type=int, default=10)  # Further reduced
    parser.add_argument("--maxQuestion", type=int, default=5)  # Further reduced
    parser.add_argument("--modelname", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--randomSeed", type=int, default=0)
    parser.add_argument("--output", type=str, default="./result_kvcache.txt")
    parser.add_argument("--kvcache", type=str, choices=["file"], default="file")
    parser.add_argument("--similarity", type=str, choices=["bertscore"], default="bertscore")
    parser.add_argument("--dataset", type=str, choices=["squad-dev"], default="squad-dev")
    parser.add_argument("--usePrompt", action="store_true")
    args = parser.parse_args()

    # Create offload directory
    os.makedirs("offload", exist_ok=True)
    
    # Clear memory before starting
    clear_gpu_memory()
    
    model_name = args.modelname
    rand_seed = args.randomSeed
    print(f"maxKnowledge {args.maxKnowledge} maxParagraph {args.maxParagraph} maxQuestion {args.maxQuestion} randomeSeed {args.randomSeed}")

    # Load model with memory optimizations
    tokenizer, model = load_model_with_memory_optimization(model_name, HF_TOKEN)
    
    kvcache_test(args)
