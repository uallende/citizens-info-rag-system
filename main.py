import streamlit as st
import weaviate
from app.rag import convert_text_to_tokens
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

def weaviate_client():
    
    client = weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
        additional_config=weaviate.config.AdditionalConfig(timeout=(60, 180)))
    return client

def main():
    st.title('Citizens information - Ask a question')

    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    max_length = 4096
    question = f"What do I do if my neighbour is having a party"

    user_input = st.text_input("Enter your question here")


    if st.button('Submit'):
        embeddings = convert_text_to_tokens(user_input)
        st.write('Embeddings:', embeddings)

if __name__ == "__main__":
    main()