from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Embedding model using SentenceTransformers (local)"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str):
        """Generate embedding for a single query string"""
        return self.model.encode([text])[0].tolist()
