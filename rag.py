#!/usr/bin/env python3
"""
RAG Benchmarking Script for Torah QA

This script benchmarks RAG performance using either CacheBlend or OpenAI API
with customizable parameters for document chunking, QPS control, and duration.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import asyncio
import random
from pathlib import Path
import re
import threading
import subprocess
from io import StringIO

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# pandas for stats analysis
import pandas as pd

# Third party imports
try:
    from transformers import AutoTokenizer
    import openai
    import numpy as np
    # Set numpy seed for reproducibility
    np.random.seed(RANDOM_SEED)
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install -r requirements.txt")
    sys.exit(1)

# Check for vllm imports (optional, only needed for CacheBlend)
VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    VLLM_AVAILABLE = True
except ImportError:
    pass

# Local imports
from faiss_utils import TorahVectorDB


@dataclass
class BenchmarkResponse:
    """Data class to store benchmark response information"""
    question: str
    correct_answer: str
    generated_answer: str
    generation_time: float
    document_chunks: List[str]
    similarity_scores: List[float]


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run"""
    base_url: Optional[str]
    use_cacheblend: bool
    model_url: str
    document_chunk_size: int
    num_documents_per_query: int
    qps: float
    duration: int
    seed: int
    warmup_questions: int
    
    # Derived fields
    output_file: str = "benchmark_results.json"


def parse_arguments() -> BenchmarkConfig:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG Benchmarking Script for Torah QA")
    
    # XOR group for base_url and cacheblend
    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument("--base-url", type=str, help="Base URL for OpenAI API")
    url_group.add_argument("--cacheblend", action="store_true", help="Use CacheBlend mode")
    
    parser.add_argument("--model-url", type=str, required=True, 
                       help="Model URL for tokenizer and inference")
    parser.add_argument("--document-chunk-size", type=int, default=512,
                       help="Document chunk size in tokens (default: 512)")
    parser.add_argument("--num-documents-per-query", type=int, default=15,
                       help="Number of documents to retrieve per query (default: 15)")
    parser.add_argument("--qps", type=float, default=4.0,
                       help="Queries per second (default: 4.0)")
    parser.add_argument("--duration", type=int, default=120,
                       help="Benchmarking duration in seconds (default: 120)")
    parser.add_argument("--output-file", type=str, default="benchmark_results.json",
                       help="Output file for results (default: benchmark_results.json)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--warmup-questions", type=int, default=30,
                       help="Number of questions to run during warmup phase (default: 30)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.cacheblend and not VLLM_AVAILABLE:
        parser.error("CacheBlend mode requires vllm and lmcache. Please install them or use --base-url instead.")
    
    # Set seeds with the provided seed value
    global RANDOM_SEED
    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    
    return BenchmarkConfig(
        base_url=args.base_url,
        use_cacheblend=args.cacheblend,
        model_url=args.model_url,
        document_chunk_size=args.document_chunk_size,
        num_documents_per_query=args.num_documents_per_query,
        qps=args.qps,
        duration=args.duration,
        seed=args.seed,
        warmup_questions=args.warmup_questions,
        output_file=args.output_file
    )


class User:
    """
    Unified interface for both CacheBlend LLM and OpenAI API client.
    Provides a common generate() method for both backends.
    """
    
    def __init__(self, config: 'BenchmarkConfig'):
        self.config = config
        self.model_url = config.model_url
        
        # Initialize tokenizer
        print(f"Loading tokenizer from {self.model_url}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_url)
        
        if config.use_cacheblend:
            self._init_cacheblend()
        else:
            self._init_openai_client()
    
    def _init_cacheblend(self):
        """Initialize CacheBlend LLM"""
        if not VLLM_AVAILABLE:
            raise RuntimeError("CacheBlend mode requires vllm and lmcache packages")
        
        print("Initializing CacheBlend LLM...")
        self._setup_cacheblend_environment()
        
        # Create KV transfer config
        ktc = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
        )
        
        # Setup LLM arguments
        llm_args = EngineArgs(
            model=self.model_url,
            kv_transfer_config=ktc,
            max_model_len=20000,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=False,
        )
        
        # Initialize LLM
        self.llm = LLM(**asdict(llm_args))
        
        # Set sampling parameters for deterministic generation
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic
            top_p=1.0,        # Deterministic
            max_tokens=100,   # Reasonable default for answers
            seed=RANDOM_SEED  # Ensure reproducibility
        )
        
        self.backend_type = "cacheblend"
        print("CacheBlend LLM initialized successfully")
    
    def _setup_cacheblend_environment(self):
        """Setup environment variables for CacheBlend"""
        # LMCache configuration
        os.environ["LMCACHE_CHUNK_SIZE"] = "256"
        os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
        os.environ["LMCACHE_BLEND_SPECIAL_STR"] = "# #"
        os.environ["LMCACHE_USE_LAYERWISE"] = "True"
        
        # Enable verbose logging for LMCache
        os.environ["LMCACHE_LOGGING_LEVEL"] = "INFO"
        os.environ["LMCACHE_VERBOSE"] = "True"
        
        if self.config.use_cacheblend:
            # Enable local CPU backend in LMCache
            os.environ["LMCACHE_LOCAL_CPU"] = "True"
            os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "50"
        else:
            os.environ["LMCACHE_LOCAL_CPU"] = "False"
    
    def _init_openai_client(self):
        """Initialize OpenAI API client"""
        print(f"Initializing OpenAI client with base URL: {self.config.base_url}")
        
        self.client = openai.OpenAI(
            base_url=self.config.base_url,
            api_key="dummy-key"  # Many local servers don't require real API keys
        )
        
        self.backend_type = "openai"
        print("OpenAI client initialized successfully")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response using the configured backend.
        
        Args:
            prompt (str): The input prompt
            
        Returns:
            str: Generated response
        """
        if self.backend_type == "cacheblend":
            return self._generate_cacheblend(prompt)
        else:
            return self._generate_openai(prompt)
    
    def _generate_cacheblend(self, prompt: str) -> str:
        """Generate using CacheBlend LLM"""
        try:
            outputs = self.llm.generate(prompts=[prompt], sampling_params=self.sampling_params)
            if outputs and outputs[0].outputs:
                return outputs[0].outputs[0].text.strip()
            return ""
        except Exception as e:
            print(f"Error in CacheBlend generation: {e}")
            return ""
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API client"""
        try:
            response = self.client.completions.create(
                model=self.model_url,
                prompt=prompt,
                max_tokens=100,
                temperature=0.0,  # Deterministic
                top_p=1.0,        # Deterministic
                seed=RANDOM_SEED, # Ensure reproducibility
                stop=None
            )
            
            if response.choices:
                return response.choices[0].text.strip()
            return ""
        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            return ""
    
    def cleanup(self):
        """Cleanup resources"""
        if self.backend_type == "cacheblend" and hasattr(self, 'llm'):
            try:
                LMCacheEngineBuilder.destroy(ENGINE_NAME)
                print("CacheBlend cleanup completed")
            except Exception as e:
                print(f"Error during cleanup: {e}")


class QPSController:
    """
    Controls query rate to maintain specified QPS (Queries Per Second).
    """
    
    def __init__(self, qps: float):
        self.qps = qps
        self.interval = 1.0 / qps if qps > 0 else 0
        self.last_request_time = 0
    
    async def wait_for_next_request(self):
        """Wait until it's time for the next request based on QPS"""
        if self.interval > 0:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.interval:
                wait_time = self.interval - time_since_last
                await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()


def load_questions() -> List[Dict[str, str]]:
    """
    Load all questions from the torah-cqa/qas directory.
    
    Returns:
        List[Dict[str, str]]: List of question-answer pairs
    """
    questions = []
    qas_dir = Path("torah-cqa/qas")
    
    if not qas_dir.exists():
        print(f"Warning: QAs directory not found: {qas_dir}")
        return questions
    
    for json_file in qas_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
                
            if isinstance(qa_data, list):
                questions.extend(qa_data)
            else:
                print(f"Warning: Unexpected format in {json_file}")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(questions)} questions from {len(list(qas_dir.glob('*.json')))} files")
    return questions


def create_rag_prompt(system_prompt: str, document_chunks: List[str], question: str, 
                     tokenizer, blend_special_str: str = "# #") -> str:
    """
    Create a RAG prompt following the CacheBlend format.
    
    Args:
        system_prompt (str): System prompt that never changes
        document_chunks (List[str]): Retrieved document chunks
        question (str): The user question
        tokenizer: Tokenizer for encoding/decoding
        blend_special_str (str): Special separator string for blending
        
    Returns:
        str: Formatted prompt ready for LLM
    """
    # Randomize document order to reduce prefix cache hits
    randomized_chunks = document_chunks.copy()
    random.shuffle(randomized_chunks)
    
    # Encode components (removing BOS token with [1:])
    sys_tokens = tokenizer.encode(system_prompt)[1:]
    blend_tokens = tokenizer.encode(blend_special_str)[1:]
    question_tokens = tokenizer.encode(question)[1:]
    
    # Build the prompt: sys_prompt + doc1 + ... + docN + question
    prompt_tokens = sys_tokens + blend_tokens
    
    for chunk in randomized_chunks:
        chunk_tokens = tokenizer.encode(chunk)[1:]
        prompt_tokens.extend(chunk_tokens)
        prompt_tokens.extend(blend_tokens)
    
    # Add the question at the end
    prompt_tokens.extend(question_tokens)
    
    # Decode back to string
    return tokenizer.decode(prompt_tokens, skip_special_tokens=True)


async def run_warmup(user, vector_db, all_questions, config, system_prompt) -> int:
    """
    Run warm-up phase to prime caches and prepare the system.
    
    Args:
        user: User object for generation
        vector_db: Vector database for document retrieval
        all_questions: List of all available questions
        config: Benchmark configuration
        system_prompt: System prompt to use
        
    Returns:
        int: Number of warmup requests completed
    """
    print(f"\n🔥 WARMUP PHASE: Running {config.warmup_questions} questions...")
    
    warmup_count = 0
    warmup_start = time.time()
    
    try:
        for i in range(config.warmup_questions):
            # Randomly select a question for warmup
            qa_pair = random.choice(all_questions)
            question = qa_pair["question"]
            
            print(f"   Warmup {i+1}/{config.warmup_questions}: {question[:40]}...")
            
            # Query vector database (same as benchmark)
            search_results = vector_db.query(question, top_k=config.num_documents_per_query)
            
            if not search_results:
                print(f"   Warning: No documents found for warmup question")
                continue
            
            # Extract document chunks
            document_chunks = [result[0] for result in search_results]
            
            # Create RAG prompt (same as benchmark)
            prompt = create_rag_prompt(
                system_prompt=system_prompt,
                document_chunks=document_chunks,
                question=question,
                tokenizer=user.tokenizer
            )
            
            # Generate response (no timing during warmup)
            user.generate(prompt)
            warmup_count += 1
            
            # Small delay to avoid overwhelming
            await asyncio.sleep(0.1)
    
    except Exception as e:
        print(f"   Error during warmup: {e}")
    
    warmup_time = time.time() - warmup_start
    print(f"   ✓ Warmup completed: {warmup_count} requests in {warmup_time:.2f}s")
    print(f"   ✓ System primed and ready for benchmark")
    
    return warmup_count


async def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Run the complete RAG benchmarking process.
    
    Args:
        config (BenchmarkConfig): Benchmark configuration
        
    Returns:
        Dict[str, Any]: Complete benchmark results
    """
    print("=" * 60)
    print("STARTING RAG BENCHMARK")
    print("=" * 60)
    print(f"Random seed: {config.seed} (for reproducibility)")
    
    # 1. Initialize User object with unified LLM interface
    print("\n1. Initializing LLM backend...")
    user = User(config)
    
    # 2. Initialize Vector Database with tokenizer
    print("\n2. Initializing Vector Database...")
    from faiss_utils import get_torah_vector_db
    vector_db = get_torah_vector_db(
        chunk_size=config.document_chunk_size, 
        max_features=10000,
        tokenizer=user.tokenizer
    )
    
    # 3. Load questions
    print("\n3. Loading questions...")
    all_questions = load_questions()
    if not all_questions:
        raise RuntimeError("No questions loaded. Please check torah-cqa/qas directory.")
    
    print(f"   Loaded {len(all_questions)} questions")
    
    # 4. Initialize QPS controller
    print(f"\n4. Setting up QPS controller for {config.qps} QPS...")
    qps_controller = QPSController(config.qps)
    
    # 5. System prompt (never changes)
    system_prompt = "You are a helpful assistant that answers questions about the Torah based on the provided context. Please provide accurate and concise answers."
    
    # 6. Run warmup phase
    if config.warmup_questions > 0:
        warmup_count = await run_warmup(user, vector_db, all_questions, config, system_prompt)
    else:
        print(f"\n🔥 WARMUP PHASE: Skipped (warmup-questions=0)")
        warmup_count = 0
    
    # 7. Run benchmark
    print(f"\n5. Running benchmark for {config.duration} seconds...")
    responses = []
    start_time = time.time()
    request_count = 0
    
    try:
        while time.time() - start_time < config.duration:
            # Wait for QPS timing
            await qps_controller.wait_for_next_request()
            
            # Randomly select a question
            qa_pair = random.choice(all_questions)
            question = qa_pair["question"]
            correct_answer = qa_pair["answer"]
            
            print(f"\n   Request {request_count + 1}: {question[:50]}...")
            
            # Query vector database
            query_start = time.time()
            search_results = vector_db.query(question, top_k=config.num_documents_per_query)
            
            if not search_results:
                print(f"   Warning: No documents found for question")
                continue
            
            # Extract document chunks and scores
            document_chunks = [result[0] for result in search_results]
            similarity_scores = [result[1] for result in search_results]
            
            # Create RAG prompt
            prompt = create_rag_prompt(
                system_prompt=system_prompt,
                document_chunks=document_chunks,
                question=question,
                tokenizer=user.tokenizer
            )
            
            # Generate response
            gen_start = time.time()
            generated_answer = user.generate(prompt)
            generation_time = time.time() - gen_start
            
            # Store response
            response = BenchmarkResponse(
                question=question,
                correct_answer=correct_answer,
                generated_answer=generated_answer,
                generation_time=generation_time,
                document_chunks=document_chunks,
                similarity_scores=similarity_scores
            )
            responses.append(response)
            
            print(f"   Generated: {generated_answer[:80]}...")
            print(f"   Generation time: {generation_time:.3f}s")
            
            request_count += 1
            
            # Check if we should continue
            if time.time() - start_time >= config.duration:
                break
    
    except KeyboardInterrupt:
        print("\n   Benchmark interrupted by user")
    except Exception as e:
        print(f"\n   Error during benchmark: {e}")
    
    finally:
        # 8. Cleanup
        print(f"\n6. Cleaning up...")
        user.cleanup()
    
    # 9. Generate results summary
    total_time = time.time() - start_time
    print(f"\n7. Generating results summary...")
    print(f"   Total requests: {len(responses)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Actual QPS: {len(responses) / total_time:.2f}")
    
    # Calculate statistics using pandas
    if responses:
        # Create DataFrame for comprehensive analysis
        df_data = []
        for i, r in enumerate(responses):
            # Add one row per response with flattened similarity scores
            avg_sim_score = sum(r.similarity_scores) / len(r.similarity_scores) if r.similarity_scores else 0
            max_sim_score = max(r.similarity_scores) if r.similarity_scores else 0
            
            df_data.append({
                'request_id': i,
                'question_length': len(r.question),
                'answer_length': len(r.generated_answer),
                'generation_time': r.generation_time,
                'num_documents': len(r.document_chunks),
                'avg_similarity_score': avg_sim_score,
                'max_similarity_score': max_sim_score,
                'total_context_length': sum(len(chunk) for chunk in r.document_chunks)
            })
        
        df = pd.DataFrame(df_data)
        
        # Calculate comprehensive statistics
        generation_stats = df['generation_time'].describe()
        similarity_stats = df['avg_similarity_score'].describe()
        
        avg_gen_time = generation_stats['mean']
        min_gen_time = generation_stats['min']
        max_gen_time = generation_stats['max']
        median_gen_time = generation_stats['50%']
        std_gen_time = generation_stats['std']
        
        avg_similarity = similarity_stats['mean']
        median_similarity = similarity_stats['50%']
        std_similarity = similarity_stats['std']
        
        print(f"   Generation Time Stats:")
        print(f"     Mean: {avg_gen_time:.3f}s, Median: {median_gen_time:.3f}s")
        print(f"     Min: {min_gen_time:.3f}s, Max: {max_gen_time:.3f}s")
        print(f"     Std Dev: {std_gen_time:.3f}s")
        print(f"   Similarity Score Stats:")
        print(f"     Mean: {avg_similarity:.4f}, Median: {median_similarity:.4f}")
        print(f"     Std Dev: {std_similarity:.4f}")
        print(f"   Context Stats:")
        print(f"     Avg context length: {df['total_context_length'].mean():.0f} chars")
        print(f"     Avg question length: {df['question_length'].mean():.0f} chars")
    
    # Prepare results dictionary with comprehensive stats
    summary_stats = {
        "total_requests": len(responses),
        "total_time": total_time,
        "actual_qps": len(responses) / total_time if total_time > 0 else 0,
    }
    
    if responses:
        # Add pandas-based statistics
        summary_stats.update({
            "generation_time": {
                "mean": float(avg_gen_time),
                "median": float(median_gen_time),
                "min": float(min_gen_time),
                "max": float(max_gen_time),
                "std": float(std_gen_time)
            },
            "similarity_score": {
                "mean": float(avg_similarity),
                "median": float(median_similarity),
                "std": float(std_similarity)
            },
            "context": {
                "avg_context_length": float(df['total_context_length'].mean()),
                "avg_question_length": float(df['question_length'].mean()),
                "avg_answer_length": float(df['answer_length'].mean()),
                "avg_documents_per_query": float(df['num_documents'].mean())
            }
        })
        
        # Include the DataFrame data for further analysis
        df_dict = df.to_dict('records')
    else:
        df_dict = []
    
    results = {
        "config": asdict(config),
        "summary": summary_stats,
        "responses": [asdict(r) for r in responses],
        "detailed_stats": df_dict,  # Include pandas DataFrame data
        "warmup_count": warmup_count  # Include warmup information
    }
    
    # Save results to file
    print(f"\n8. Saving results to {config.output_file}...")
    with open(config.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBenchmark completed! Results saved to {config.output_file}")
    return results


def main():
    """Main function"""
    config = parse_arguments()
    print(f"Starting RAG benchmark with config: {asdict(config)}")
    
    # Run the benchmark
    try:
        results = asyncio.run(run_benchmark(config))
        print(f"\nBenchmark Summary:")
        print(f"  Warmup Requests: {results['warmup_count']}")
        print(f"  Total Requests: {results['summary']['total_requests']}")
        print(f"  Actual QPS: {results['summary']['actual_qps']:.2f}")
        
        if 'generation_time' in results['summary']:
            gen_stats = results['summary']['generation_time']
            sim_stats = results['summary']['similarity_score']
            print(f"  AVERAGE Generation Time: {gen_stats['mean']:.3f}s (±{gen_stats['std']:.3f}s)")
            print(f"  Similarity Score: {sim_stats['mean']:.4f} (±{sim_stats['std']:.4f})")
        else:
            print("  No statistics available (no responses generated)")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 