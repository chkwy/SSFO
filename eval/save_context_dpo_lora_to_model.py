import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.utils import GenerationConfig


def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    # base.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    lora_path = "Bibaolong/Context-Faithful-LLaMA-3-8b-instruct"
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    output = "/home/YiWang/DFO-script/save_models/context-dpo-llama3-8b-instruct"

    apply_lora(model_path, output, lora_path)
