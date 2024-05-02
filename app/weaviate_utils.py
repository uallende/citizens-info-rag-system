import weaviate
import streamlit as st
from weaviate.classes.query import MetadataQuery
from constants import WEAVIATE_COLLECTION_NAME
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def initialise(client):
    collection = client.collections.get(WEAVIATE_COLLECTION_NAME)
    print(f'Collection retreived from Weaviate')
    return collection

def check_data_in_db(client):
    print("Checking if data is in database...")
    collection_name = WEAVIATE_COLLECTION_NAME
    print("Data is in database:", client.collections.exists(collection_name))
    return client.collections.exists(collection_name)

def load_weaviate_client(host, port, grpc_port, secure=False):
    connection_params = weaviate.connect.ConnectionParams.from_params(
        http_host=host,
        http_port=port,
        http_secure=secure,
        grpc_host=host,
        grpc_port=grpc_port,
        grpc_secure=secure,
    )
    client = weaviate.WeaviateClient(connection_params)
    client.connect()
    return client

'''
THIS SECTION WORKS LOCALLY BUT NOT WHEN IMAGES ARE LOADED ON A DOCKER CONTAINER
DOCKER DEPLOYMENT REQUIRES A CUSTOM CONNECTION AND NOT A LOCAL ONE
client = weaviate.connect_to_local(
port=8080,
grpc_port=50051,
additional_config=weaviate.config.AdditionalConfig(timeout=(60, 180))
)
'''

def load_weaviate_local_connection(port, grpc_port):
    client = weaviate.connect_to_local(
    port=port,
    grpc_port=grpc_port,
    additional_config=weaviate.config.AdditionalConfig(timeout=(60, 180))
    )
    client.connect()
    logger.info(f"Client connected to weaviate")
    return client

def retrieve_nearest_content(collection, query_embeddings):
    logger.info(f"Collection: {collection}")
    logger.info(f"query_embeddings: {query_embeddings}")
    response = collection.query.near_vector(
        near_vector=query_embeddings.tolist(),  
        target_vector='default', 
        return_properties=['body', 'title'],
        limit=2,
        return_metadata=MetadataQuery(distance=True)
    )
    return response