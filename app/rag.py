import weaviate
import weaviate.classes as wvc
import os
import torch.nn.functional as F
import torch, gc
from dotenv import load_dotenv

from weaviate.classes.query import MetadataQuery
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from torch import Tensor



def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def convert_text_to_tokens(text:str, model, tokenizer, max_length):

    batch_dict = tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to('cuda')
    output = model(**batch_dict)
    embeddings = last_token_pool(output.last_hidden_state, batch_dict['attention_mask'])[0].float().cpu().detach().numpy()
    return embeddings

# openai_api_key = os.getenv("OPENAI_KEY")

# client = weaviate.connect_to_local(
#     port=8080,
#     grpc_port=50051,
#     additional_config=weaviate.config.AdditionalConfig(timeout=(60, 180)),
#     headers={
#         "X-OpenAI-Api-Key": openai_api_key  
#     }
# )