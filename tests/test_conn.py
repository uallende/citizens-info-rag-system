import weaviate
import os


# config.py
HOST = "weaviate"  # or "localhost" for local development
PORT = "8080"
GRPC_PORT = "50051"
SECURE = False

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

def load_weaviate_local_connection(port, grpc_port):
    client = weaviate.connect_to_local(
    port=port,
    grpc_port=grpc_port,
    additional_config=weaviate.config.AdditionalConfig(timeout=(60, 180))
    )
    client.connect()
    return client

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

client = load_weaviate_local_connection(PORT, GRPC_PORT)
# client = load_weaviate_client(host=HOST, 
#                                   port=PORT, 
#                                   grpc_port=GRPC_PORT, 
#                                   secure=SECURE)
client.connect()
print(f'Connection established!')
