---


# KVCache for Context-Aware Generation (CAG)

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Why KVCache for CAG?](#why-kvcache-for-cag)
- [Datasets](#datasets)
- [Deployment Instructions](#deploymentinstructions)
- [Usage in CAG Applications](#usage-in-cag-applications)
  - [Basic Integration with Attention Mechanisms](#basic-integration-with-attention-mechanisms)
  - [Multihead Attention Caching](#multihead-attention-caching)
  - [Managing Context Window](#managing-context-window)
- [API Reference](#api-reference)
- [Thread Safety for Parallel Processing](#thread-safety-for-parallel-processing)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)

---

## 🧠 Overview

KVCache is an efficient in-memory key-value store specifically optimized for use in Context-Aware Generation (CAG) architectures employing multihead attention. This implementation provides the underlying caching infrastructure to efficiently store and retrieve key-value pairs used in attention mechanisms, enabling faster processing and reduced memory footprint during inference.

---

## 🚀 Features

- Optimized for storage of attention key-value pairs
- Support for multihead attention caching patterns
- Optional expiration time for cached items
- Automatic clearing of expired items
- Thread-safe operations
- Configurable maximum cache size
- Performance stats tracking (hits, misses, etc.)

---

## ❓ Why KVCache for CAG?

Context-Aware Generation models, particularly those using transformer architectures with multihead attention, benefit from efficient key-value caching:

- ✅ **Reduced Computation**: Avoid redundant attention calculations
- ⚡ **Faster Inference**: Quick retrieval of cached context
- 📦 **Memory Efficiency**: Manage memory footprint with expiration and limits

---

## Datasets
<p><a href="https://www.kaggle.com/api/v1/datasets/download/stanfordu/stanford-question-answering-dataset" target="_blank">SQuAD</a></p>
<p><a href="https://www.kaggle.com/api/v1/datasets/download/jeromeblanchet/hotpotqa-question-answering-dataset" target="_blank">HotpotQA</a></p>
<p><a href="https://ai.google.com/research/NaturalQuestions/download" target="_blank">Google Natural Questions</a></p>
<p><a href="https://huggingface.co/datasets/mandarjoshi/trivia_qa" target="_blank">TriviaqQA</a></p>

## 📦 Deployment guidelines for CAG:
Update your API keys in the environment variables

To run CAG:
python kvcache.py --dataset "trivia-qa" --modelname "meta-llama/Llama-2-7b-chat-hf" --randomSeed 0 --output "./result_kvcache.txt"
You can change the dataset and the llama models

To run RAG:
python rag.py \
    --index "openai" \           # can change to other ai providers based on API keys
    --dataset "trivia-qa" \    # can be changed based on the dataset
    --modelname "meta-llama/Llama-2-7b-chat-hf" \ # model name can be changed
    --topk 3 \                 # Number of top documents to retrieve
    --maxKnowledge 10 \        # Maximum number of knowledge items
    --maxParagraph 100 \       # Maximum paragraphs per knowledge item
    --maxQuestion 50 \         # Maximum number of questions to process
    --output "./rag_results.txt"

---

## 🧩 Usage in CAG Applications

### 📌 Basic Integration with Attention Mechanisms

python
from kvcache import KVCache

# Create a new cache configured for attention mechanism storage
attention_cache = KVCache(max_size=10000)

# Store key-value pairs from an attention head
attention_cache.set("layer1_head0_k", key_tensor)
attention_cache.set("layer1_head0_v", value_tensor)

# Retrieve cached attention keys/values during inference
cached_key = attention_cache.get("layer1_head0_k")
cached_value = attention_cache.get("layer1_head0_v")


### 🧠 Multihead Attention Caching

python
for head in range(num_heads):
    attention_cache.set(f"layer{layer}_head{head}_k", key_tensors[head])
    attention_cache.set(f"layer{layer}_head{head}_v", value_tensors[head])

for head in range(num_heads):
    k = attention_cache.get(f"layer{layer}_head{head}_k")
    v = attention_cache.get(f"layer{layer}_head{head}_v")
    # Use k, v in attention computation


### 🕰 Managing Context Window

python
# Set a value that expires after a defined time window
attention_cache.set("token_123_context", context_data, expiry_time=context_window_size)


---

## 📘 API Reference

### `KVCache` Class

#### Constructor

python
KVCache(max_size=10000, cleanup_interval=60)


- `max_size` (int): Maximum number of items the cache can hold
- `cleanup_interval` (int): Interval in seconds for automatic cleanup of expired items

#### Methods

- `get(key, default=None)`
- `set(key, value, expiry_time=None)`
- `delete(key)`
- `clear()`
- `cleanup()`
- `reset_stats()`
- `size()`
- `keys()`
- `values()`
- `items()`

### Cache Statistics (via `attention_cache.stats`)

- `hits`
- `misses`
- `insertions`
- `deletions`
- `expirations`

---

## 🧵 Thread Safety for Parallel Processing

All operations on the `KVCache` are thread-safe, using internal locks to ensure consistency across concurrent reads/writes — ideal for multi-head attention in parallel environments.

---


## 📈 Performance Considerations

- Optimized for O(1) lookup times
- Tune `max_size` for your model’s token window × attention heads
- For large-scale applications, consider **hierarchical cache layers**

---



