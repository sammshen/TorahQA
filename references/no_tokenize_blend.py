# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False, blend_special_str: str = "# #"
):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 40GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "40"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=20000,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )

    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: '# #')",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    setup_environment_variables(args.use_disk, args.blend_special_str)

    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # This example script runs two requests with a shared prefix.
        # Define the shared prompt and specific prompts
        sys_prompt = tokenizer.encode("You are a very helpful assistant.")
        chunk1_prompt = tokenizer.encode("Hello, how are you?" * 250)
        chunk2_prompt = tokenizer.encode("Hello, what's up?" * 250)
        chunk3_prompt = tokenizer.encode("Hi hi hi hi hi hi" * 250)
        chunk4_prompt = tokenizer.encode("Bye bye bye bye bye bye" * 250)
        blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))
        first_prompt = tokenizer.decode(
            sys_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + chunk3_prompt
            + blend_special_str
            + chunk4_prompt
            + blend_special_str
            + tokenizer.encode("Hello, my name is")
        )

        second_prompt = tokenizer.decode(
            sys_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            + chunk3_prompt
            + blend_special_str
            + chunk4_prompt
            + blend_special_str
            + tokenizer.encode("Hello, how are you?")
        )

        third_prompt = tokenizer.decode(
            sys_prompt
            + blend_special_str
            + chunk4_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + chunk3_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            + tokenizer.encode("Hello, my name is")
        )

        fourth_prompt = tokenizer.decode(
            sys_prompt
            + blend_special_str
            + chunk3_prompt
            + blend_special_str
            + chunk2_prompt
            + blend_special_str
            + chunk4_prompt
            + blend_special_str
            + chunk1_prompt
            + blend_special_str
            + tokenizer.encode("Hello, my name is")
        )

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

        # Print the first output
        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        # print the second output
        print_output(llm, second_prompt, sampling_params, "second")

        time.sleep(1)

        # print the third output
        print_output(llm, third_prompt, sampling_params, "third")

        time.sleep(1)

        # print the fourth output
        print_output(llm, fourth_prompt, sampling_params, "fourth")

    
if __name__ == "__main__":
    main()
