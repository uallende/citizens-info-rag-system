# main.py
import streamlit as st
import os
from rag import generate_final_answer, parse_llm_generated_answer, generate_lightweight_embeddings
from weaviate_utils import load_weaviate_client, load_weaviate_local_connection
from weaviate_utils import check_data_in_db, initialise, retrieve_nearest_content
from init_data import initialise_data
from config import HOST, PORT, GRPC_PORT, SECURE

def main():
    client = load_weaviate_client(host=HOST, 
                                  port=PORT, 
                                  grpc_port=GRPC_PORT, 
                                  secure=SECURE)
    
    if not check_data_in_db(client):
        print("Current working directory:", os.getcwd())
        initialise_data()

    collection = initialise(client)

    st.title('Citizens Information - Ask a question')
    user_input = st.text_input("Enter your question here")

    if st.button('Submit'):
        # query_embeddings = generate_text_embeddings(text=user_input)
        query_embeddings = generate_lightweight_embeddings(text=user_input)
        response = retrieve_nearest_content(collection, query_embeddings)
        
        for o in response.objects:
            context = o.properties['body']
            break

        llm_answer = generate_final_answer(context, user_input)
        parsed_user_answer = parse_llm_generated_answer(llm_answer)
        st.markdown(f'{parsed_user_answer} \n \n')
        st.markdown(f'Original article: \n {context}')

if __name__ == "__main__":
    main()