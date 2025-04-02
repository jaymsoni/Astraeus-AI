"""Embedding utilities for document search."""
from typing import List, Dict, Any
import google.generativeai as genai
import numpy as np
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
genai.configure(api_key=api_key)

# Define the embedding model to use
EMBEDDING_MODEL = "models/embedding-001"

class EmbeddingManager:
    def __init__(self):
        """
        Initialize the embedding manager.
        """
        self.embeddings = {}
        self.documents = {}
        
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a document and its embedding to the manager.
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata about the document
        """
        try:
            # Generate embedding for the document using Gemini
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=content,
                task_type="RETRIEVAL_DOCUMENT"  # Specify task type for document embedding
            )
            embedding = np.array(result['embedding'])
            
            # Store the embedding and document info
            self.embeddings[doc_id] = embedding
            self.documents[doc_id] = {
                'content': content,
                'metadata': metadata or {}
            }
            
        except Exception as e:
            print(f"Error generating embedding for document {doc_id}: {str(e)}")
            # Decide if you want to re-raise or handle gracefully
            # raise
        
    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document and its embedding.
        
        Args:
            doc_id: ID of the document to remove
        """
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        if doc_id in self.documents:
            del self.documents[doc_id]
            
    def search(self, query: str, k: int = 4, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Search for documents using semantic similarity.
        
        Args:
            query: Search query
            k: Maximum number of results to return
            threshold: Minimum similarity score required for a result to be included
            
        Returns:
            List of documents with similarity scores above the threshold
        """
        if not self.embeddings:
            return []
            
        try:
            # Generate embedding for the query
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="RETRIEVAL_QUERY"  # Specify task type for query embedding
            )
            query_embedding = np.array(result['embedding'])
            
            # Calculate cosine similarity between query and all documents
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                # Ensure embeddings are 1D arrays
                if doc_embedding.ndim > 1:
                    doc_embedding = doc_embedding.flatten()
                if query_embedding.ndim > 1:
                    query_embedding = query_embedding.flatten()
                    
                # Check for zero vectors
                norm_query = np.linalg.norm(query_embedding)
                norm_doc = np.linalg.norm(doc_embedding)
                
                if norm_query == 0 or norm_doc == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(query_embedding, doc_embedding) / (norm_query * norm_doc)
                    
                similarities.append((doc_id, similarity))
                
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by threshold and limit to k results
            results = []
            for doc_id, score in similarities:
                if score >= threshold and len(results) < k:
                    doc = self.documents[doc_id]
                    results.append({
                        'id': doc_id,
                        'content': doc['content'], # Consider returning snippets later
                        'metadata': doc['metadata'],
                        'score': float(score)
                    })
                elif len(results) >= k:
                    # Stop adding results once we reach k items above threshold
                    break
                
            return results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
        
    def save_embeddings(self, save_path: str) -> None:
        """
        Save embeddings and document info to disk.
        
        Args:
            save_path: Path to save the embeddings
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = save_dir / 'embeddings.json'
        try:
            with open(embeddings_file, 'w') as f:
                json.dump({
                    doc_id: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    for doc_id, embedding in self.embeddings.items()
                }, f)
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")
            
        # Save document info
        docs_file = save_dir / 'documents.json'
        try:
            with open(docs_file, 'w') as f:
                json.dump(self.documents, f)
        except Exception as e:
            print(f"Error saving documents metadata: {str(e)}")
            
    def load_embeddings(self, load_path: str) -> None:
        """
        Load embeddings and document info from disk.
        
        Args:
            load_path: Path to load the embeddings from
        """
        load_dir = Path(load_path)
        
        # Load embeddings
        embeddings_file = load_dir / 'embeddings.json'
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'r') as f:
                    embeddings_data = json.load(f)
                    self.embeddings = {
                        doc_id: np.array(embedding)
                        for doc_id, embedding in embeddings_data.items()
                    }
            except Exception as e:
                print(f"Error loading embeddings: {str(e)}")
                
        # Load document info
        docs_file = load_dir / 'documents.json'
        if docs_file.exists():
            try:
                with open(docs_file, 'r') as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Error loading documents metadata: {str(e)}") 