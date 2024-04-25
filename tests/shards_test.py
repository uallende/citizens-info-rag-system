from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, AutoTokenizer, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
import torch
import os

DEVICE='cuda'

hf_token = os.getenv("HUGGINGFACE_TOKEN")
model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

config = AutoConfig.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    token=hf_token
)
config.max_position_embeddings = 8096

quantization_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=True,
    token=hf_token
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map=DEVICE,
    offload_folder="./offload",  # specify an existing folder for offload
    token=hf_token
)

print(model.config)
print(tokenizer.model_info)