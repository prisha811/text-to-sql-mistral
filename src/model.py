import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "models/adapter"


def load_model():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base Mistral 7B...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()

    print("Model loaded successfully!")

    return model, tokenizer