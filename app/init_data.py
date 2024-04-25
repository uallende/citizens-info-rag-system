import os
from dotenv import load_dotenv
from weaviate.classes.config import Property, DataType
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import *
import weaviate


def load_pdf_documents(path_to_pdf):
    documents_text = []
    for doc in os.listdir(path_to_pdf):
        doc_path = f'{path_to_pdf}/{doc}'
        loader = PyPDFLoader(doc_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=2500//2)
        docs = text_splitter.split_documents(pages)
        documents_text.append(docs)
    return [item for sublist in documents_text for item in sublist]

def extract_document_data(documents_text):
    document_objs = []
    for d in documents_text:
        title = d.metadata['source']
        page = str(d.metadata['page'])  # page number to string
        body = d.page_content
        document_objs.append({
            "page": page,
            "title": title,
            "body": body
        })
    return document_objs

def create_weaviate_collection(client, collection_name):
    try:
        # Create a new collection
        client.collections.create(
            "citizens_info_docs",

            properties=[  
                Property(name="page", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="body", data_type=DataType.TEXT),
            ]
        )
        print(f"Collection '{collection_name}' has been created successfully.")
    except weaviate.exceptions.WeaviateBaseError as e:
        if isinstance(e, weaviate.exceptions.ObjectAlreadyExistsError):
            print(f"Collection '{collection_name}' already exists. Skipping creation.")
        elif isinstance(e, weaviate.exceptions.UnexpectedStatusCodeError) and e.status_code == 422:
            print(f"Collection '{collection_name}' already exists. Skipping creation.")
        else:
            raise

    return client.collections.get(collection_name)

def populate_weaviate_collection(collection, document_objs, body_vectors):
    with collection.batch.dynamic() as batch:
        for i, data_row in enumerate(document_objs):
            batch.add_object(
                properties=data_row,
                vector=body_vectors[i].tolist(),
            )

def initialise_data(client):
    collection_name = "citizens_info_docs"
    if not client.collections.exists(collection_name):
        print("Collection does not exist, creating and populating...")
        path_to_pdf = 'pdf_docs'
        documents_text = load_pdf_documents(path_to_pdf)

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        body_vectors = []
        for d in documents_text:
            body = d.page_content
            embeddings = model.encode(body)
            body_vectors.append(embeddings)

        document_objs = extract_document_data(documents_text)

        collection = create_weaviate_collection(client, collection_name)
        populate_weaviate_collection(collection, document_objs, body_vectors)

        print(f'Vector database has been populated with pdf info')
    else:
        print(f"Collection '{collection_name}' already exists, skipping data loading...")

if __name__ == "__main__":
    initialise_data()

