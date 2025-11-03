import google.generativeai as genai
from typing import List, Dict
import os

class GeminiAnswerGenerator:
    def __init__(self, model_name="gemini-2.5-flash", api_key: str = None):
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "No relevant context found for this query."

        context = "\n\n".join([chunk["text"] for chunk in context_chunks])

        prompt = f"""
You are an AI legal assistant. Use ONLY the provided context to answer the question.

Context:
{context}

Question: {query}

Rules:
1. Cite relevant sections or Acts if mentioned.
2. Be factual and concise.
3. If context is insufficient, clearly state so.
"""

        response = self.model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else str(response)
