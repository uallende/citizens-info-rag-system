import torch, gc, os
import streamlit as st

from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
from torch import Tensor
from constants import *
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def generate_text_embeddings(text:str, hf_token):
    max_length = 4096
    model = load_embeddings_model(hf_token)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    batch_dict = tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    model.eval()
    output = model(**batch_dict)

    with torch.no_grad():
        embeddings = last_token_pool(output.last_hidden_state, batch_dict['attention_mask'])[0].float().cpu().detach().numpy()
    clear_gpu_memory(model, tokenizer)
    return embeddings

def clear_gpu_memory(model, tokenizer):
    model.cpu()
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

def load_embeddings_model(hf_token):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModel.from_pretrained(
        'Salesforce/SFR-Embedding-Mistral',
        trust_remote_code=True,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        token = hf_token
    )

    return model

def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_model_from_local(model_folder):
    print("Loading model from local directory...")
    config = AutoConfig.from_pretrained(model_folder)
    model = AutoModelForCausalLM.from_pretrained(model_folder, config=config)
    return model

def load_model_from_pretrained(model_name_or_path, hf_token, device):
    print("Loading model from pretrained...")
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        token=hf_token
    )

    config.max_position_embeddings = MAX_POS_EMBEDDINGS

    quantization_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device,
        token=hf_token
    )
    return model

def load_tokenizer_from_local(tokenizer_folder, model_folder):
    print("Loading tokenizer from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder, config=AutoConfig.from_pretrained(model_folder))
    return tokenizer

def load_tokenizer_from_pretrained(model_name_or_path, hf_token):
    print("Loading tokenizer from pretrained...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        token=hf_token
    )
    return tokenizer

@st.cache_resource(show_spinner=False)
def load_llm(model_folder, tokenizer_folder, model_name_or_path, hf_token, device):
    create_folder_if_not_exists(model_folder)
    create_folder_if_not_exists(tokenizer_folder)

    model_config_file = os.path.join(model_folder, "config.json")
    model_generation_config_file = os.path.join(model_folder, "generation_config.json")
    model_file = os.path.join(model_folder, "model.safetensors")
    tokenizer_config_file = os.path.join(tokenizer_folder, "tokenizer_config.json")

    if os.path.exists(model_config_file) and os.path.exists(model_generation_config_file) and os.path.exists(model_file):
        model = load_model_from_local(model_folder)
    else:
        model = load_model_from_pretrained(model_name_or_path, hf_token, device)
        model.save_pretrained(model_folder)

    if os.path.exists(tokenizer_config_file):
        tokenizer = load_tokenizer_from_local(tokenizer_folder, model_folder)
    else:
        tokenizer = load_tokenizer_from_pretrained(model_name_or_path, hf_token)
        tokenizer.save_pretrained(tokenizer_folder)

    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

def generate_final_answer(context, user_input, model_folder, tokenizer_folder, model_name_or_path, hf_token, device):
    model, tokenizer = load_llm(model_folder, tokenizer_folder, model_name_or_path, hf_token, device)
    prompt = [
        {
            "role": "user", 
            "content": (
                f"Based on the following context {context}, "
                f"can you provide an answer to this {user_input}. "
                f"the answer should be only reflect facts that are present in the context."
                f"If the information is not clear say I don't know but don't make up any information"
            ),
        }
    ]
    
    logger.info(f"Full prompt: {prompt}")
    
    encoded_prompt = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(DEVICE)
    logger.info(f"Tokenizer output: {encoded_prompt}")
    logger.info(f"Number of tokens: {encoded_prompt['input_ids'].shape[1]}")
    
    generated_ids = model.generate(encoded_prompt, max_new_tokens=2500, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded

def parse_llm_generated_answer(decoded):
    # Extract the answer after the [/INST] token
    start_token = "[/INST]"
    start_index = decoded[0].find(start_token)

    if start_index != -1:
        start_index += len(start_token)
        answer = decoded[0][start_index:].strip().replace("</s>", "")

    else:
        return None
    return answer

def lightweight_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def generate_lightweight_embeddings(text:str):
    model = lightweight_embedding_model()
    embeddings = model.encode(text)
    return embeddings
