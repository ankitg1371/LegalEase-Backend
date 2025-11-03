from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import LegalRetriever
from generator import GeminiAnswerGenerator
from dotenv import load_dotenv
import os

# 🔹 Load environment variables
load_dotenv()

app = FastAPI(title="Legal RAG Backend (Gemini)", version="1.3")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Initialize components
retriever = LegalRetriever(
    persist_dir="./legal_chroma_db",
    collection_name="legal_documents"
)

generator = GeminiAnswerGenerator(
    api_key=os.getenv("GEMINI_API_KEY")  # load key from .env
)

# Request body model
class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"status": "✅ Legal RAG API running with Gemini and .env loaded."}


@app.post("/query")
def query_legal_docs(request: QueryRequest):
    """Main RAG endpoint — expects JSON body like: { "query": "your question" }"""
    try:
        query = request.query.strip()
        print(f"🔍 Query received: {query}")
        print(f"📦 Collection count: {retriever.collection.count()}")

        chunks = retriever.retrieve(query, top_k=5)
        print(f"📄 Retrieved {len(chunks)} chunks")

        answer = generator.generate(query, chunks)

        return {
            "query": query,
            "answer": answer,
            "sources": chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
