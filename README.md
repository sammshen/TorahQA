# RAG Benchmarking

Benchmarks RAG performance using CacheBlend or OpenAI API with Torah QA data.

## Installation

```bash
pip install -r requirements.txt
```

For CacheBlend mode: `pip install vllm lmcache`

## Usage

**OpenAI API:**
```bash
# First serve the vllm http server (or any other baseline)
vllm serve "mistralai/Mistral-7B-Instruct-v0.2"
python rag.py --base-url "http://localhost:8000/v1" --model-url "mistralai/Mistral-7B-Instruct-v0.2" --output-file "vanilla_vllm_results.json"
```

**CacheBlend:**
```bash
python rag.py --cacheblend --model-url "mistralai/Mistral-7B-Instruct-v0.2" --output-file "cacheblend_results.json"
```

**With custom settings:**
```bash
python rag.py --cacheblend --model-url "mistralai/Mistral-7B-Instruct-v0.2" \
    --num-documents-per-query 5 --warmup-questions 50 --duration 60 --qps 2.0 \
    --output-file "your_baseline_benchmark.json"
```

## Key Arguments

- `--base-url` / `--cacheblend`: Choose backend (mutually exclusive)
- `--model-url`: Model for tokenizer and inference (required)
- `--document-chunk-size`: Chunk size in tokens (default: 512)
- `--num-documents-per-query`: Number of document chunks to retrieve per query (default: 15)
- `--qps`: Queries per second (default: 4.0)
- `--duration`: Test duration in seconds (default: 120)
- `--seed`: Random seed for reproducibility (default: 42)
- `--warmup-questions`: Number of warmup questions to run (default: 30)
- `--output-file`: Output filename for results (default: benchmark_results.json)

## Features

- **Unified backends**: CacheBlend (vLLM + LMCache) and OpenAI API compatibility
- **Smart document retrieval**: FAISS vector search with TF-IDF embeddings, token-based chunking, and sentence boundary detection
- **Realistic RAG simulation**: Document randomization, QPS control, and warm-up phase
- **Comprehensive analysis**: Pandas-based statistics with generation times, similarity scores, and detailed metrics

## How It Works

1. **Loads** Torah documents and QA pairs from `torah-cqa/`
2. **Builds** FAISS index with TF-IDF embeddings for semantic search  
3. **Retrieves** top-k relevant chunks for each question
4. **Constructs** prompts with system text + shuffled documents + question using `"# #"` separators
5. **Generates** answers with controlled QPS and deterministic parameters
6. **Analyzes** performance with comprehensive statistics

## Data & Output

**Input**: Torah documents (`torah-cqa/contexts/`) and QA pairs (`torah-cqa/qas/`)

**Output**: JSON results with comprehensive statistics including:
- **Per-Request Data**: Question, generated answer, correct answer, generation time, document chunks, similarity scores
- **Generation Performance**: Mean, median, min, max, and standard deviation of generation times
- **Retrieval Quality**: Similarity score statistics and context analysis
- **System Metrics**: Actual QPS achieved, total requests processed, warmup phase statistics
- **Detailed Analytics**: Pandas DataFrame with per-request metrics for further analysis

## Example Results (on H100 80GB) with example above:

### Huge decrease in generation time with minimal loss in accuracy

vanilla vllm:
```text
Benchmark Summary:
  Warmup Requests: 30
  Total Requests: 5
  Actual QPS: 0.07
  Average Generation Time: 14.643s (±28.981s)
  Similarity Score: 0.1226 (±0.0545)
```

cacheblend: 
```text
Benchmark Summary:
  Warmup Requests: 30
  Total Requests: 62
  Actual QPS: 1.90
  Average Generation Time: 0.312s (±1.768s)
  Similarity Score: 0.1209 (±0.0612)
```

