# vector_store.py
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use sentence-transformers for embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create collection if it doesn't exist
        try:
            self.collection = self.client.get_collection("documents")
        except ValueError:
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=sentence_transformer_ef
            )
    
    def extract_text_from_pdf(self, pdf_path: str, chunk_size: int = 1000) -> List[str]:
        """Extract text from PDF and split into chunks"""
        reader = PdfReader(pdf_path)
        text_chunks = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Split into paragraphs
                paragraphs = text.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(para.strip()) < 20:  # Skip very short paragraphs
                        continue
                        
                    if len(current_chunk) + len(para) > chunk_size:
                        if current_chunk:
                            text_chunks.append(current_chunk.strip())
                        current_chunk = para
                    else:
                        current_chunk += " " + para if current_chunk else para
                
                if current_chunk:
                    text_chunks.append(current_chunk.strip())
        
        return text_chunks
    
    def add_pdf(self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process PDF, extract text, and store in vector DB"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        filename = os.path.basename(pdf_path)
        doc_id = f"doc_{filename.replace('.', '_')}"
        
        # Extract text chunks
        text_chunks = self.extract_text_from_pdf(pdf_path)
        
        # Add each chunk to the collection
        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Prepare metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "source": filename,
                "chunk_id": i,
                "doc_id": doc_id
            })
            
            # Add to collection
            self.collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[chunk_id]
            )
        
        return doc_id
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents given a query"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = []
        for i, doc in enumerate(results['documents'][0]):
            documents.append({
                "content": doc,
                "metadata": results['metadatas'][0][i] if 'metadatas' in results else {},
                "id": results['ids'][0][i]
            })
        
        return documents
