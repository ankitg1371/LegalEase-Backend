from typing import List, Dict
from embeddings.embedder import EmbeddingGenerator
from db.chroma_client import get_chroma_client, get_collection

class LegalRetriever:
    """Semantic retriever over ChromaDB"""
    
    def __init__(self, persist_dir="./legal_chroma_db", collection_name="legal_documents"):
        self.embedder = EmbeddingGenerator()
        self.client = get_chroma_client(persist_dir)
        self.collection = get_collection(self.client, collection_name)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return top chunks for a query"""
        query_embedding = self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i]
            })
        return documents
