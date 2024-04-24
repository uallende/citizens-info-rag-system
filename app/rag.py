import weaviate
import torch, gc
import streamlit as st
import os

from weaviate.classes.query import MetadataQuery
from weaviate.connect import ConnectionParams
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
from torch import Tensor
from constants import *
from sentence_transformers import SentenceTransformer


# def weaviate_client():
#     client = weaviate.connect_to_local(
#         port=8080,
#         grpc_port=50051,
#         additional_config=weaviate.config.AdditionalConfig(timeout=(60, 180)))
#     return client

def weaviate_client():
    
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host="0.0.0.0",  # Use 0.0.0.0 to allow connections from outside the container
            http_port="8080",     # The port exposed by the Weaviate container
            http_secure=False,
            grpc_host="0.0.0.0",
            grpc_port="50051",
            grpc_secure=False,
        ),
    )
    return client


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def generate_text_embeddings(text:str):
    max_length = 4096
    model = load_embeddings_model()
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    batch_dict = tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    model.eval()
    output = model(**batch_dict)

    with torch.no_grad():
        embeddings = last_token_pool(output.last_hidden_state, batch_dict['attention_mask'])[0].float().cpu().detach().numpy()
    clear_gpu_memory(model, tokenizer)
    return embeddings

def retrieve_nearest_content(collection, query_embeddings):
    response = collection.query.near_vector(
        near_vector=query_embeddings.tolist(),  
        target_vector='default', 
        return_properties=['body', 'title'],
        limit=2,
        return_metadata=MetadataQuery(distance=True)
    )
    return response

def clear_gpu_memory(model, tokenizer):
    model.cpu()
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

def load_embeddings_model():
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
        quantization_config=bnb_config
    )

    return model

@st.cache_resource
def load_llm():
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.max_position_embeddings = 8096

    quantization_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=DEVICE,
        offload_folder="./offload"
    )

    return model, tokenizer

def generate_final_answer(context, user_input):
    model, tokenizer = load_llm()
    prompt = [
        {

        "role": "user", 
        "content": (
            f"Based on the following context {context}, "
            f"can you provide an answer to this {user_input}. "
            f"the answer should be only reflect facts that are present in the context."
            f"If the information is not clear say I don't know but don't make up any information"),

         }
            ]   
    
    encoded_prompt = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(encoded_prompt, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)  # Use encoded_prompt directly
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

@st.cache_resource
def generate_lightweight_embeddings(text:str):
    model = lightweight_embedding_model()
    embeddings = model.encode(text)
    return embeddings
