# Cache-Augmented Generation (CAG)

<img src="https://github.com/hhhuang/CAG/blob/main/overview.png?raw=true" width=600 alt="Overview of CAG">

Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for enhancing language models by integrating external knowledge sources. However, RAG also introduces several challenges, including:  
- **Retrieval Latency** – Delays caused by real-time retrieval steps.  
- **Retrieval Errors** – Inaccuracies in selecting relevant documents.  
- **System Complexity** – Increased architectural and maintenance overhead.  

To address these limitations, we propose **Cache-Augmented Generation (CAG)**—an alternative paradigm that bypasses real-time retrieval. CAG leverages the extended context windows of modern large language models (LLMs) by preloading all relevant resources into the model’s context and caching its runtime parameters. During inference, the preloaded KV-cache enables the model to generate responses directly, eliminating the need for retrieval.  

**Advantages of CAG**  
- **Reduced Latency** – Eliminates real-time retrieval, enabling faster inference.  
- **Improved Reliability** – Minimizes retrieval errors while maintaining context relevance.  
- **Simplified Design** – Provides a streamlined, retrieval-free alternative to RAG, achieving comparable or superior results with lower complexity.  

**Limitations of CAG**  
- **Limited Knowledge Size** – CAG requires the entire knowledge source to fit within the context window, making it less suitable for tasks involving extremely large datasets.  
- **Context Length Constraints** – The performance of LLMs may degrade with very long contexts ([reference](https://arxiv.org/pdf/2404.02060v2)).  

Our [paper](https://arxiv.org/abs/2412.15605), which will be presented at [the ACM Web Conference 2025](https://www2025.thewebconf.org/) as a short paper, investigates the relationship between model performance and context length, providing insights into scenarios where CAG excels.  

The limitations of CAG are rapidly being addressed by advancements in LLMs with longer context windows and improved capabilities for extracting relevant information from extended inputs. As these models continue to evolve, CAG is expected to handle increasingly complex applications, making it a practical and scalable alternative to traditional RAG.  

---

## Installation 
```bash
pip install -r ./requirements.txt
```

## Preparation
> [!IMPORTANT]  
> download the required `squad` and `hotpotqa` datasets by curl script
> ```bash
> sh ./downloads.sh
> ```

> [!IMPORTANT]
> create `.env` file by `.env.template` and input the keys required
> ```bash
> cp ./.env.template ./.env
> ```

## Usage
- `rag.py` is for RAG Experiment
- `kvcache.py` is for CAG Experiment

## Parameter Usage -- kvcache.py
- `--kvcache`: "file"
- `--dataset`: "hotpotqa-train" or "squad-train"
- `--similarity` "bertscore"
- `--modelname`: "meta-llama/Llama-3.1-8B-Instruct"
- `--maxKnowledge`: "", int, select how many document in dataset, explanation in Note
- `--maxParagraph`: 100
- `--maxQuestion` int, max question number, explanation in Note
- `--randomSeed`: "", int, a random seed number
- `--output`: "", str, output filepath string
- `--usePrompt`, add this parameter if not using CAG knowledge cache acceleration 

### Example -- kvcache.py
```bash
python ./kvcache.py --kvcache file --dataset "squad-train" --similarity bertscore \
    --maxKnowledge 5 --maxParagraph 100 --maxQuestion 1000  \
    --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed 0 \
    --output "./result_kvcache.txt"
```

## Parameter Usage -- rag.py
- `--index`: "openai" or "bm25"
- `--dataset`: "hotpotqa-train" or "squad-train"
- `--similarity` "bertscore"
- `--maxKnowledge`: "", int, select how many document in dataset, explanation in Note
- `--maxParagraph`: 100
- `--maxQuestion` int, max question number, explanation in Note
- `--topk`: int, the similarity topk of retrieval
- `--modelname`: "meta-llama/Llama-3.1-8B-Instruct"
- `--randomSeed`: "", int, a random seed number
- `--output`: "", str, output filepath string

### Example -- rag.py
```bash
python ./rag.py --index "bm25" --dataset "hotpotqa-train" --similarity bertscore \
    --maxKnowledge 80 --maxParagraph 100 --maxQuestion 80 --topk 3 \
    --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed  0 \
    --output  "./rag_results.txt"
```

### Note:
#### `--maxKnowledge` parameter notice: 
> [!NOTE]
> Approximate Tokens count corresponding to knowledge document size of "squad-train" and "hotpotqa-train" dataset. 

> datasets=("squad-train")
> - when k = 3, tokens = 21,000
> - when k = 4, tokens = 32,000
> - when k = 7, tokens = 50,000
> 
> datasets=("hotpotqa-train")
> - all k = 7405 article, tokens = 10,038,084 
> - when k = 1, tokens = 1,400
> - when k = 16, tokens = 22,400
> - when k = 24, tokens = 33,667
> - when k = 32, tokens = 44,800
> - when k = 48, tokens = 64,000
> - when k = 64, tokens = 85,000
> - when k = 80, tokens = 106,000

#### `--maxQuestion` parameter notice:
> - when using "squad-train" dataset, 1 knowledge has average 150 questions
> - when using "hotpotqa-train" dataset, 1 knowledge has 1 question

> [!TIP]
> Since 1 document in "hotpoqa-train" dataset has only 1 question, it may not satisfy large-scale evaluation.
> Multiple evaluation could be a relatively better approach.
> 

#### `Docker`

To build the docker image, run
```bash
 docker build -t my-cag-app .
 ```

 and to run the container, run this for GPU users

```bash
docker run --gpus all -it --rm my-cag-app
```
OR
```bash
docker run -it --rm my-cag-app
```
for CPU users.

if the .env file details were empty while building you will get error similar to this below

```bash
Traceback (most recent call last):
  File "/app/./kvcache.py", line 35, in <module>
    env = validate_env_variables()
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/./kvcache.py", line 31, in validate_env_variables
    raise ValueError(f"Missing required environment variable: {key}")
ValueError: Missing required environment variable: HF_TOKEN
``` 

so ensure you populate the `.env` file before building the docker image 

Note that the he `CMD` directive in the Dockerfile runs the `kvcache.py ` script by default. You can override this in the docker run command if you'd like to execute other scripts like rag.py. For example:

```bash
docker run --gpus all -it --rm my-cag-app python ./rag.py --index "bm25" --dataset "hotpotqa-train" --similarity bertscore --maxKnowledge 80 --maxParagraph 100 --maxQuestion 80 --topk 3 --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed 0 --output "./rag_results.txt"
```

## Citation
```
@misc{chan2024dontragcacheaugmentedgeneration,
      title={Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks}, 
      author={Brian J Chan and Chao-Ting Chen and Jui-Hung Cheng and Hen-Hsen Huang},
      year={2024},
      eprint={2412.15605},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.15605}, 
}
```

## Acknowledgments
This work was partially supported by National Science and Technology Council (NSTC), Taiwan, under the grant 112-2221-E-001-016-MY3, by Academia Sinica, under the grant 236d-1120205, and by National Center for High-performance Computing (NCHC), National Applied Research Laboratories (NARLabs), and NSTC under the project "Taiwan's 113th year endeavoring in the promotion of a trustworthy generative AI large language model and the cultivation of literacy capabilities (Trustworthy AI Dialog Engine, TAIDE)".
We sincerely thank [Discover AI](https://www.youtube.com/watch?v=NaEf_uiFX6o) and the many individuals who have introduced, shared, and discussed our work, contributing to its broader visibility and impact.
