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
python rag_benchmark.py --base-url "http://localhost:8000/v1" --model-url "mistralai/Mistral-7B-Instruct-v0.2"
```

**CacheBlend:**
```bash
python rag_benchmark.py --cacheblend --model-url "mistralai/Mistral-7B-Instruct-v0.2" --lmcache
```

## Key Arguments

- `--base-url` / `--cacheblend`: Choose backend (mutually exclusive)
- `--model-url`: Model for tokenizer and inference (required)
- `--document-chunk-size`: Chunk size in tokens (default: 512)
- `--qps`: Queries per second (default: 4.0)
- `--duration`: Test duration in seconds (default: 30)
- `--seed`: Random seed for reproducibility (default: 42)

## Features

- Unified interface for CacheBlend and OpenAI API backends
- Token-based document chunking with precise control
- QPS-controlled benchmarking with reproducible results (seeded)
- Comprehensive pandas-based statistical analysis
- FAISS vector search with TF-IDF embeddings

## Data & Output

**Input**: Torah documents (`torah-cqa/contexts/`) and QA pairs (`torah-cqa/qas/`)

**Output**: JSON results with comprehensive statistics including generation times, similarity scores, and detailed per-request data. 