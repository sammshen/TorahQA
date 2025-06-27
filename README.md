# Jewish RAG Benchmarking

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

- Unified interface for CacheBlend and OpenAI API backends
- Token-based document chunking with precise control
- QPS-controlled benchmarking with reproducible results (seeded)
- **Warm-up phase**: Primes caches before benchmarking for accurate measurements
- Comprehensive pandas-based statistical analysis
- **FAISS vector search with TF-IDF embeddings**:
  - **Document Processing**: Automatically loads all `.txt` files from `torah-cqa/contexts/`
  - **Smart Chunking**: Documents are split using token-based chunking (with provided tokenizer) or character-based chunking, with sentence boundary detection to avoid breaking mid-sentence
  - **TF-IDF Vectorization**: Uses scikit-learn's `TfidfVectorizer` with unigrams and bigrams, English stop words removal, and configurable max features (default: 10,000)
  - **FAISS Indexing**: Employs `IndexFlatIP` (Inner Product) for fast cosine similarity search after L2 normalization
  - **Singleton Pattern**: Database is built once and reused across all queries for efficiency
  - **Retrieval**: For each question, returns top-k most semantically similar document chunks with similarity scores
  - **Integration**: Retrieved chunks are automatically formatted into RAG prompts using the CacheBlend-compatible format with special separators

## RAG Pipeline Workflow

**End-to-End Process for Each Question:**

1. **Question Selection**: Randomly selects a question from `torah-cqa/qas/` JSON files
2. **Document Retrieval**: 
   - Vectorizes the question using the same TF-IDF model used for documents
   - Performs cosine similarity search against the FAISS index
   - Returns top-k most relevant document chunks (default: 15 chunks)
   - Each chunk includes similarity score and metadata (source file, position)
3. **Prompt Construction**:
   - **System Prompt**: Fixed instructional text for consistent behavior
   - **Document Randomization**: Shuffles retrieved chunks to reduce prefix rigidity and match real world RAG more closely
   - **Token Encoding**: Uses model tokenizer to encode each component separately
   - **CacheBlend Format**: Assembles prompt as: `system + sep + doc1 + sep + doc2 + ... + sep + question`
   - **Special Separators**: Uses `"# #"` tokens between components for cache blending
4. **LLM Generation**:
   - **CacheBlend Mode**: Uses vLLM with KV cache transfer and layerwise blending
   - **OpenAI Mode**: Standard completions API with temperature=0 for deterministic results
   - **Generation Parameters**: Fixed seed, controlled max tokens, deterministic sampling
5. **Response Collection**: Records generation time, answer text, and retrieval metadata
6. **Statistical Analysis**: Pandas-based analysis of generation times, similarity scores, and answer lengths

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

