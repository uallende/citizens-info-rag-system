from constants import WEAVIATE_COLLECTION_NAME

@st.cache_resource
def initialise(client):
    collection = client.collections.get(WEAVIATE_COLLECTION_NAME)
    print(f'collection retreived from weaviate')
    return collection

def check_data_in_db(client):
    collection_name = WEAVIATE_COLLECTION_NAME
    return client.collections.exists(collection_name)
