DEVICE = 'cuda'
WEAVIATE_COLLECTION_NAME = "citizens_info_docs"
MODEL_SAVE_DIRECTORY = 'model'
TOKENIZER_SAVE_DIRECTORY = 'tokenizer'
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MODEL_NAME = "NousResearch/Hermes-2-Pro-Llama-3-8B"
MAX_POS_EMBEDDINGS = 8096
CHUNK_SIZE = MAX_POS_EMBEDDINGS * (2/3)