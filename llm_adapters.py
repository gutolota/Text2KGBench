# File: llm_adapters.py
# Place this file in the root of your forked text2kgbench repository.
# This module defines BaseLLM, OllamaLLM, HFLLM and a factory to obtain the desired adapter.

from abc import ABC, abstractmethod

# If using Ollama via LangChain
from langchain_community.llms import Ollama

# If using HuggingFace locally
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class BaseLLM(ABC):
    """
    Abstract base class for LLM adapters.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Given a prompt string, returns the model's generated text.
        """
        pass

class OllamaLLM(BaseLLM):
    """
    Adapter for Ollama models via LangChain.
    Usage:
        llm = OllamaLLM(model_name="llama3")
        out = llm.generate("Your prompt here.")
    """

    def __init__(self, model_name: str, temperature: float = 0.0):
        # Adjust parameters as supported by your Ollama install
        self.llm = Ollama(model=model_name)
        self.temperature = temperature

    def generate(self, prompt: str, max_tokens: int = 1024, **kwargs) -> str:
        # Some Ollama wrappers support optional kwargs like max_tokens
        return self.llm.invoke(
            prompt,
            temperature=self.temperature,
            max_tokens=max_tokens,
            **kwargs
        )

class HFLLM(BaseLLM):
    """
    Adapter for HuggingFace-hosted models (e.g. Gemma 3n).
    Ensure you have sufficient GPU or CPU resources.
    Usage:
        llm = HFLLM(model_name="google/gemma-3n")
        out = llm.generate("Your prompt here.")
    """

    def __init__(self, model_name: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        # Create a text-generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False
        )

    def generate(self, prompt: str, max_length: int = 2048, **kwargs) -> str:
        # Note: passing do_sample=False for deterministic outputs
        outputs = self.pipeline(
            prompt,
            max_length=max_length,
            do_sample=False,
            **kwargs
        )
        return outputs[0]["generated_text"]


def get_llm_adapter(model_key: str) -> BaseLLM:
    """
    Factory to return the correct LLM adapter.

    model_key options:
      - "ollama_llama3"  => Llama 3 via Ollama
      - "ollama_gemma3"  => Gemma 3 via Ollama
      - "hf_gemma3n"     => Gemma 3n via HuggingFace
    """
    if model_key == "ollama_llama3":
        return OllamaLLM(model_name="llama3")
    if model_key == "ollama_gemma3":
        return OllamaLLM(model_name="gemma3")
    if model_key == "hf_gemma3n":
        return HFLLM(model_name="google/gemma-3n")
    raise ValueError(f"Unknown model key: {model_key}")


# -----------------------------------------------------------------------------
# File: run_benchmark.py
# Place this file in the root of your text2kgbench repo and invoke via
#   python run_benchmark.py --model MODEL_KEY --output_dir ./results

import argparse
import os
from llm_adapters import get_llm_adapter
# Import the main benchmark runner from text2kgbench
# Adjust this import to match the real entry point of the benchmark
from text2kgbench.runner import run_full_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run text2kgbench with different LLM adapters."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key to use: ollama_llama3 | ollama_gemma3 | hf_gemma3n"
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory to save outputs and metrics."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Instantiate the adapter
    llm = get_llm_adapter(args.model)

    # Run the full benchmark: datasets, prompts, eval
    # The function signature may vary: adapt as needed
    metrics = run_full_benchmark(
        llm=llm,
        output_dir=args.output_dir,
        prompt_templates_dir="./prompts",
        datasets_dir="./datasets",
        temperature=args.temperature
    )

    # Save a summary CSV or JSON
    summary_path = os.path.join(args.output_dir, f"summary_{args.model}.json")
    with open(summary_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    print(f"Benchmark completed for {args.model}. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
