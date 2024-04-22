import streamlit as st
from app.rag import weaviate_client, retrieve_nearest_content
from app.rag import generate_final_answer, parse_llm_generated_answer, generate_lightweight_embeddings

@st.cache_resource
def initialise():
    client = weaviate_client()
    collection = client.collections.get("citizens_info_docs")
    return collection

def main():

    st.title('Citizens Information - Ask a question')
    user_input = st.text_input("Enter your question here")
    collection = initialise()

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