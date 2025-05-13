# rag.py
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from ollama_client import OllamaModel

class RAG:
    def __init__(self, vector_store: VectorStore, model: OllamaModel):
        self.vector_store = vector_store
        self.model = model
    
    def query(self, user_query: str, n_docs: int = 3) -> str:
        # 1. Retrieve relevant documents
        retrieved_docs = self.vector_store.search(user_query, n_results=n_docs)
        
        # 2. Format context from retrieved documents
        context = self._format_context(retrieved_docs)
        
        # 3. Generate answer with context
        prompt = self._create_rag_prompt(user_query, context)
        
        # 4. Generate response
        response = self.model.generate(prompt)
        
        return response
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        context_parts = []
        
        for i, doc in enumerate(documents):
            source_info = f"Source: {doc['metadata'].get('source', 'Unknown')}" if 'metadata' in doc else "Source: Unknown"
            context_parts.append(f"[Document {i+1}]\n{doc['content']}\n{source_info}\n")
        
        return "\n".join(context_parts)
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        return f"""
You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

USER QUESTION:
{query}

Please answer the question based only on the provided context. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer this question."

ANSWER:
"""
