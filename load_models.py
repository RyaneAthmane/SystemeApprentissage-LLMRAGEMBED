# ce code fournit des fonctions pour charger différents types de modèles
# de langage naturel, y compris des modèles quantifiés et des modèles complets.
# Les modèles peuvent être chargés à l'aide de différentes classes et options,
# en fonction du type de modèle et du type de périphérique spécifié.

import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from constants import CONTEXT_WINDOW_SIZE, MAX_NEW_TOKENS, N_GPU_LAYERS, N_BATCH, MODELS_PATH


def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging):

    try:
        logging.info("Using Llamacpp for GGUF/GGML quantized models")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
            "n_batch": N_BATCH,
        }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 1
        if device_type.lower() == "cuda":
            kwargs["n_gpu_layers"] = N_GPU_LAYERS

        return LlamaCpp(**kwargs)
    except:
        return None


def load_full_model(model_id, model_basename, device_type, logging):
    if device_type.lower() in ["mps", "cpu"]:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(
            model_id, cache_dir="./models/")
        model = LlamaForCausalLM.from_pretrained(
            model_id, cache_dir="./models/")
    else:
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir="./models/")
        logging.info("Tokenizer loaded")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=MODELS_PATH,
        )
        model.tie_weights()
    return model, tokenizer
