---


# KVCache for Cache Augmented Generation (CAG)

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Why KVCache for CAG?](#why-kvcache-for-cag)
- [Datasets](#datasets)
- [Deployment Instructions](#deployment-instructions)
- [Code Reference](#code-reference)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)

---

## Overview

KVCache is an efficient in-memory key-value store specifically optimized for use in Cache Augmented Generation (CAG) architectures employing multihead attention. This implementation provides the underlying caching infrastructure to efficiently store and retrieve key-value pairs used in attention mechanisms, enabling faster processing and reduced memory footprint during inference.

---

## Features

- Optimized for storage of attention key-value pairs
- Support for multihead attention caching patterns
- Optional expiration time for cached items
- Automatic clearing of expired items
- Thread-safe operations
- Configurable maximum cache size
- Performance stats tracking (hits, misses, etc.)

---

## Why KVCache for CAG?

Cache Augmented Generation models, particularly those using transformer architectures with multihead attention, benefit from efficient key-value caching:

- ✅ **Reduced Computation**: Avoid redundant attention calculations
- ⚡ **Faster Inference**: Quick retrieval of cached context
- 📦 **Memory Efficiency**: Manage memory footprint with expiration and limits

---

## Datasets
<p><a href="https://www.kaggle.com/api/v1/datasets/download/stanfordu/stanford-question-answering-dataset" target="_blank">SQuAD</a></p>
<p><a href="https://www.kaggle.com/api/v1/datasets/download/jeromeblanchet/hotpotqa-question-answering-dataset" target="_blank">HotpotQA</a></p>
<p><a href="https://ai.google.com/research/NaturalQuestions/download" target="_blank">Google Natural Questions</a></p>
<p><a href="https://huggingface.co/datasets/mandarjoshi/trivia_qa" target="_blank">TriviaqQA</a></p>

## Deployment Instructions
Update your API keys in the environment variables

To run CAG:
```python
kvcache.py --dataset "trivia-qa" --modelname "meta-llama/Llama-2-7b-chat-hf" --randomSeed 0 --output "./result_kvcache.txt"
```
You can change the dataset and the llama models

To run RAG:
```python
rag.py \
    --index "openai" \           # can change to other ai providers based on API keys
    --dataset "trivia-qa" \    # can be changed based on the dataset
    --modelname "meta-llama/Llama-2-7b-chat-hf" \ # model name can be changed
    --topk 3 \                 # Number of top documents to retrieve
    --maxKnowledge 10 \        # Maximum number of knowledge items
    --maxParagraph 100 \       # Maximum paragraphs per knowledge item
    --maxQuestion 50 \         # Maximum number of questions to process
    --output "./rag_results.txt"
```

---

## Code Reference

### `KVCache` Class

#### Constructor

```python
KVCache(max_size=10000, cleanup_interval=60)
```

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


## Performance Considerations

- Optimized for O(1) lookup times
- Tune `max_size` for your model's token window × attention heads
- For large-scale applications, consider **hierarchical cache layers**

---

## Contributing
<p><a href="https://github.com/deepikasai-mettu" target="_blank">Deepika Mettu</a></p>

<p><a href="https://github.com/Sachiprogrammer" target="_blank">Sachi Patel</a></p>

<p><a href="https://github.com/ruju0901" target="_blank">Ruju Shah</a></p>

---
