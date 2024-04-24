# main.py
import streamlit as st
from rag import load_weaviate_client, load_weaviate_local_connection, retrieve_nearest_content
from rag import generate_final_answer, parse_llm_generated_answer, generate_lightweight_embeddings
from init_data import initialise_data
from config import HOST, PORT, GRPC_PORT, SECURE
from constants import WEAVIATE_COLLECTION_NAME

def create_weaviate_client(host, port, grpc_port, secure):
    return load_weaviate_client(host=host, port=port, grpc_port=grpc_port, secure=secure)

@st.cache_resource
def initialise(client):
    collection = client.collections.get(WEAVIATE_COLLECTION_NAME)
    print(f'collection retreived from weaviate')
    return collection

def check_data_in_db(client):
    collection_name = WEAVIATE_COLLECTION_NAME
    return client.collections.exists(collection_name)

def main():
    client = load_weaviate_client(host=HOST, 
                                  port=PORT, 
                                  grpc_port=GRPC_PORT, 
                                  secure=SECURE)
    
    if not check_data_in_db(client):
        initialise_data()
        
    st.title('Citizens Information - Ask a question')
    user_input = st.text_input("Enter your question here")
    collection = initialise(client)

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