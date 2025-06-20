
import os
from abc import ABC, abstractmethod
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LLMClient(ABC):
    """Abstract base class for a generic LLM client."""
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the response.

        :param prompt: The input prompt for the language model.
        :return: The response from the language model as a string.
        """
        pass

class OllamaClient(LLMClient):
    """
    A client for interacting with Large Language Models hosted by Ollama.
    """
    def __init__(self, model_name: str):
        """
        Initializes the Ollama client.

        :param model_name: The name of the model to use from Ollama (e.g., 'llama3.1').
        """
        try:
            self.llm = Ollama(model=model_name)
            print(f"Ollama client initialized for model: {model_name}")
        except Exception as e:
            print(f"Error initializing Ollama client: {e}")
            raise

    def get_response(self, prompt: str) -> str:
        """
        Gets a response from the Ollama-hosted model.

        :param prompt: The input prompt.
        :return: The model's response.
        """
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            print(f"An error occurred while getting response from Ollama: {e}")
            return f"Error: {e}"

class HuggingFaceClient(LLMClient):
    """
    A client for interacting with Large Language Models from the Hugging Face Hub.
    This client runs the model locally.
    """
    def __init__(self, model_id: str):
        """
        Initializes the Hugging Face client.

        :param model_id: The Hugging Face model ID (e.g., 'google/gemma-3n-E4B-it-litert-preview').
        """
        # Ensure you have a Hugging Face token set up in your environment.
        # You can log in via the terminal: `huggingface-cli login`
        if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
            print("Warning: HUGGING_FACE_HUB_TOKEN environment variable not set.")
            print("You might encounter issues downloading gated models.")

        try:
            print(f"Initializing Hugging Face client for model: {model_id}")
            
            # Use 4-bit quantization to reduce memory usage
            bnb_config = None
            if torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto", # Automatically use GPU if available
                quantization_config=bnb_config
            )

            # Create a transformers pipeline
            text_generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512, # Adjust as needed
                pad_token_id=tokenizer.eos_token_id
            )

            self.llm_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)
            print("Hugging Face client initialized successfully.")

        except Exception as e:
            print(f"Error initializing Hugging Face client: {e}")
            print("Please ensure you have accepted the model's license on the Hugging Face Hub.")
            raise

    def get_response(self, prompt: str) -> str:
        """
        Gets a response from the locally-run Hugging Face model.

        :param prompt: The input prompt.
        :return: The model's response.
        """
        try:
            return self.llm_pipeline.invoke(prompt)
        except Exception as e:
            print(f"An error occurred while getting response from Hugging Face model: {e}")
            return f"Error: {e}"

def get_llm_client(model_name: str) -> LLMClient:
    """
    Factory function to get the appropriate LLM client based on the model name.
    """
    if model_name.lower() in ['llama3.1', 'gemma3']:
        # Map friendly names to actual Ollama model tags
        ollama_model_map = {
            'llama3.1': 'llama3.1',
            'gemma3': 'gemma:9b' # Example tag, adjust if needed
        }
        return OllamaClient(model_name=ollama_model_map[model_name])
    elif model_name.lower() == 'gemma3n':
        # Use the specific Hugging Face model ID for Gemma 3n
        return HuggingFaceClient(model_id='google/gemma-3n-E4B-it-litert-preview')
    else:
        raise ValueError(f"Unknown model name: {model_name}")