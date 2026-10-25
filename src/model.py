"""
model.py - model loading helpers.

Key design point: the base model and the adapter are loaded SEPARATELY.

    PeftModel.from_pretrained(base_model, adapter) modifies `base_model` IN PLACE.
    After that call, `base_model` also contains the LoRA layers, so it is no longer
    a clean base model. That is why load_base_model() and attach_adapter() are two
    steps, and why generate_sql.py finishes all base-model generation BEFORE it
    attaches the adapter.

Quantization settings match training (LLM.ipynb): 4-bit NF4, double quantization,
float16 compute dtype.

If you see "bitsandbytes is not available" in Colab/Kaggle, restart the runtime
after `pip install` and run again.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = ROOT / "models" / "adapter"


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_causal_lm():
    kwargs = dict(
        quantization_config=get_bnb_config(),
        device_map="auto",
    )
    try:
        # Newer transformers versions call this argument `dtype`.
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, **kwargs)
    except TypeError:
        # Older versions call it `torch_dtype`.
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, **kwargs)


def load_base_model(for_training: bool = False):
    """Clean 4-bit base Mistral 7B, with NO adapter."""
    print(f"Loading base model {BASE_MODEL} in 4-bit (NF4)...")
    model = _load_causal_lm()
    if not for_training:
        model.eval()
    return model


def attach_adapter(base_model, adapter_path=ADAPTER_PATH):
    """
    Attach the trained QLoRA adapter to `base_model`.
    WARNING: this modifies `base_model` in place; do not use it as a "base" model afterwards.
    """
    from peft import PeftModel

    adapter_path = Path(adapter_path)
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_path / required).exists():
            raise FileNotFoundError(
                f"{adapter_path / required} not found. "
                "Copy your trained adapter into models/adapter/ (or pass --adapter)."
            )

    print(f"Attaching adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    return model


def adapter_fingerprint(adapter_path=ADAPTER_PATH) -> str:
    """First 12 hex characters of the SHA-256 of adapter_model.safetensors: records WHICH adapter produced a result."""
    import hashlib

    digest = hashlib.sha256()
    with open(Path(adapter_path) / "adapter_model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def count_lora_layers(model) -> int:
    """Number of LoRA-wrapped linear layers. For r=16 on q/k/v/o_proj in 32 layers this is 128."""
    return sum(1 for name, _ in model.named_modules() if name.endswith("lora_A"))


def load_finetuned():
    """Convenience loader for apps: returns (fine-tuned model, tokenizer)."""
    tokenizer = load_tokenizer()
    model = attach_adapter(load_base_model())
    return model, tokenizer
