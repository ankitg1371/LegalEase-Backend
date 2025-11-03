import chromadb
from chromadb.config import Settings

def get_chroma_client(persist_dir="./legal_chroma_db"):
    """Return initialized ChromaDB client"""
    return chromadb.PersistentClient(persist_dir)

def get_collection(client, name="legal_documents"):
    """Return Chroma collection"""
    return client.get_or_create_collection(name=name)
