import streamlit as st
import weaviate, os, torch
from dotenv import load_dotenv
from app.rag import generate_text_embeddings, load_embeddings_model
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

def weaviate_client():
    
    client = weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
        additional_config=weaviate.config.AdditionalConfig(timeout=(60, 180)))
    return client

def main():
    st.title('Citizens Information - Ask a question')

    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    os.chdir(parent_dir)

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    model = load_embeddings_model()

    max_length = 4096
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    user_input = st.text_input("Enter your question here")


    if st.button('Submit'):
        embeddings = generate_text_embeddings(text=user_input, 
                                            model=model, 
                                            tokenizer=tokenizer, 
                                            max_length=max_length)
        
        st.write('Embeddings:', embeddings)

if __name__ == "__main__":
    main()