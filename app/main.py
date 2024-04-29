import streamlit as st
import os
from rag import generate_final_answer, parse_llm_generated_answer, generate_lightweight_embeddings
from weaviate_utils import load_weaviate_client, load_weaviate_local_connection
from weaviate_utils import check_data_in_db, initialise, retrieve_nearest_content
from init_data import initialise_data
from config import HOST, PORT, GRPC_PORT, SECURE
from constants import MODEL_SAVE_DIRECTORY, TOKENIZER_SAVE_DIRECTORY, LLM_MODEL_NAME, DEVICE
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

def main():
    client = load_weaviate_client(host=HOST, 
                                  port=PORT, 
                                  grpc_port=GRPC_PORT, 
                                  secure=SECURE)
    
    # client = load_weaviate_local_connection(port=PORT, grpc_port=GRPC_PORT) # ONLY FOR LOCAL TESTING

    if not check_data_in_db(client):
        print("Current working directory:", os.getcwd())
        initialise_data(client)

    collection = initialise(client)

    st.title("Citizens Information - Ask a question", anchor="title")
    st.write("")  # Add some whitespace

    user_input = st.text_input("Enter your question here")

    if st.button("Submit"):
        with st.spinner("Converting query to text..."):
            query_embeddings = generate_lightweight_embeddings(text=user_input)
            response = retrieve_nearest_content(collection, query_embeddings)
            
            for o in response.objects:
                context = o.properties['body']
                break

        with st.spinner("Generating a response..."):
            llm_answer = generate_final_answer(context, 
                                            user_input, 
                                            MODEL_SAVE_DIRECTORY, 
                                            TOKENIZER_SAVE_DIRECTORY, 
                                            LLM_MODEL_NAME, 
                                            hf_token, 
                                            DEVICE)
            parsed_user_answer = parse_llm_generated_answer(llm_answer)
        
        st.subheader("Answer")
        st.markdown(f"{parsed_user_answer} \n ")
        st.subheader("Original context")
        st.markdown(f"{context}")

if __name__ == "__main__":
    main()