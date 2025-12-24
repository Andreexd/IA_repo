import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LORA_PATH = "../models/lora"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False
    )

    device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None,
        low_cpu_mem_usage=False
    )

    model.to(device)

    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
        device_map=None
    )

    model.eval()
    return tokenizer, model


def infer(prompt, tokenizer, model):
    template = f"""
### Instrucción:
{prompt}

### Respuesta:
"""

    inputs = tokenizer(template, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.005,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Respuesta:" in result:
        result = result.split("### Respuesta:")[1].strip()
    
    if "<|end|>" in result:
        result = result.split("<|end|>")[0].strip()
    if "### Rol:" in result:
        result = result.split("### Rol:")[0].strip()
    if "### Instrucción:" in result:
        result = result.split("### Instrucción:")[0].strip()
    
    sentences = result.split('. ')
    if len(sentences) > 2:
        result = '. '.join(sentences[:2]) + '.'

    return result


def main():
    tokenizer, model = load_model()

    while True:
        user_input = input("Tú: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            break

        respuesta = infer(user_input, tokenizer, model)
        print("\IA:\n", respuesta, "\n")


if __name__ == "__main__":
    main()
