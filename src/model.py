import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, DEVICE

def load_model():
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        ).to(DEVICE)

    return model, tokenizer
