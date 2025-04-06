"""Embedding utilities for document search."""
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai
import numpy as np
from pathlib import Path
import json
import os
import logging
from dotenv import load_dotenv
import re
from collections import Counter

# Import the new chunker and reranker
from backend.utils.chunker import DocumentChunker
from backend.utils.reranker import SearchReranker

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
genai.configure(api_key=api_key)

# Define the embedding model to use
EMBEDDING_MODEL = "models/embedding-001"

class BM25Retriever:
    """
    BM25 retrieval algorithm for keyword-based document search.
    Implements the Okapi BM25 ranking function to score documents based on term frequency.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the BM25 retriever.
        
        Args:
            k1: Term saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs = Counter()  # Document frequencies of terms
        self.doc_lengths = {}       # Lengths of each document
        self.avg_doc_length = 0     # Average document length
        self.total_docs = 0         # Total number of documents
        self.inverted_index = {}    # Inverted index: term -> list of documents
        self.documents = {}         # Document storage: doc_id -> tokenized content
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms for BM25 scoring.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, split by non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
        
    def add_document(self, doc_id: str, content: str) -> None:
        """
        Add a document to the BM25 index.
        
        Args:
            doc_id: Document identifier
            content: Document content
        """
        # Tokenize document
        tokens = self.tokenize(content)
        
        if not tokens:
            return
            
        # Update document storage
        self.documents[doc_id] = tokens
        
        # Update document length
        self.doc_lengths[doc_id] = len(tokens)
        
        # Update term frequencies
        term_freqs = Counter(tokens)
        
        # Update inverted index and document frequencies
        for term in term_freqs:
            if term not in self.inverted_index:
                self.inverted_index[term] = set()
            self.inverted_index[term].add(doc_id)
            self.doc_freqs[term] += 1
            
        # Update total docs and recalculate average length
        self.total_docs = len(self.documents)
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
    
    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the BM25 index.
        
        Args:
            doc_id: Document identifier
        """
        if doc_id not in self.documents:
            return
            
        # Get the document tokens
        tokens = self.documents[doc_id]
        
        # Update document frequencies and inverted index
        for term in set(tokens):
            if term in self.inverted_index and doc_id in self.inverted_index[term]:
                self.inverted_index[term].remove(doc_id)
                self.doc_freqs[term] -= 1
                
                # Clean up empty entries
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
                if self.doc_freqs[term] == 0:
                    del self.doc_freqs[term]
        
        # Remove document length
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]
            
        # Remove from document storage
        del self.documents[doc_id]
        
        # Update total docs and recalculate average length
        self.total_docs = len(self.documents)
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25 scoring.
        
        Args:
            query: Search query
            k: Maximum number of results to return
            
        Returns:
            List of document IDs with BM25 scores
        """
        # Tokenize query
        query_terms = self.tokenize(query)
        
        if not query_terms or self.total_docs == 0:
            return []
            
        # Calculate scores for each document
        scores = {}
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
                
            # Calculate inverse document frequency (IDF)
            idf = np.log(1 + (self.total_docs - self.doc_freqs[term] + 0.5) / 
                             (self.doc_freqs[term] + 0.5))
            
            # Score each document containing this term
            for doc_id in self.inverted_index[term]:
                if doc_id not in scores:
                    scores[doc_id] = 0
                    
                # Calculate term frequency in this document
                term_freq = self.documents[doc_id].count(term)
                
                # BM25 formula components
                doc_length = self.doc_lengths[doc_id]
                length_norm = (1 - self.b) + self.b * (doc_length / self.avg_doc_length)
                term_score = idf * ((term_freq * (self.k1 + 1)) / 
                                   (term_freq + self.k1 * length_norm))
                
                scores[doc_id] += term_score
        
        # Sort by score and limit to k results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Format results
        results = []
        for doc_id, score in sorted_results:
            results.append({
                'id': doc_id,
                'score': float(score),
                'bm25_score': float(score),  # Keep original BM25 score
                'retrieval_method': 'bm25'
            })
            
        return results

class EmbeddingManager:
    def __init__(self, use_chunks: bool = True, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the embedding manager.
        
        Args:
            use_chunks: Whether to split documents into chunks
            chunk_size: Size of chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.embeddings = {}
        self.documents = {}
        self.parent_chunks = {}  # Maps parent docs to their chunks
        
        # Initialize chunker and reranker
        self.use_chunks = use_chunks
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.reranker = SearchReranker(api_key=api_key)
        
        # Initialize BM25 retriever for sparse search
        self.bm25_retriever = BM25Retriever()
        
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a document and its embedding to the manager.
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata about the document
        """
        if not content:
            logger.warning(f"Empty content for document: {doc_id}")
            return
            
        # Store the full document (regardless of chunking)
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {}
        }
        
        # Add the full document to BM25 retriever
        self.bm25_retriever.add_document(doc_id, content)
        
        try:
            if self.use_chunks and len(content) > self.chunker.chunk_size:
                # Split document into chunks and process each
                chunks = self.chunker.chunk_document(doc_id, content, metadata)
                
                # Track chunks belonging to this document
                self.parent_chunks[doc_id] = [chunk["id"] for chunk in chunks]
                
                # Process each chunk
                for chunk in chunks:
                    chunk_id = chunk["id"]
                    chunk_content = chunk["content"]
                    chunk_metadata = chunk["metadata"]
                    
                    # Generate embedding for the chunk
                    self._generate_embedding(chunk_id, chunk_content)
                    
                    # Add chunk to BM25 retriever
                    self.bm25_retriever.add_document(chunk_id, chunk_content)
                    
                    # Store the chunk info
                    self.documents[chunk_id] = {
                        'content': chunk_content,
                        'metadata': chunk_metadata
                    }
                
                logger.info(f"Added document '{doc_id}' with {len(chunks)} chunks")
            else:
                # Generate embedding for the entire document
                self._generate_embedding(doc_id, content)
                logger.info(f"Added document '{doc_id}' as a single unit")
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {str(e)}")
    
    def _generate_embedding(self, doc_id: str, content: str) -> None:
        """
        Generate and store an embedding for content.
        
        Args:
            doc_id: ID to associate with the embedding
            content: Text content to embed
        """
        try:
            # Generate embedding using Gemini
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=content,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding = np.array(result['embedding'])
            
            # Store the embedding
            self.embeddings[doc_id] = embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for {doc_id}: {str(e)}")
            
    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document and its embedding.
        
        Args:
            doc_id: ID of the document to remove
        """
        # Check if this is a parent document with chunks
        if doc_id in self.parent_chunks:
            # Remove all associated chunks
            for chunk_id in self.parent_chunks[doc_id]:
                if chunk_id in self.embeddings:
                    del self.embeddings[chunk_id]
                if chunk_id in self.documents:
                    del self.documents[chunk_id]
                # Remove from BM25 retriever
                self.bm25_retriever.remove_document(chunk_id)
            
            # Remove the chunks mapping
            del self.parent_chunks[doc_id]
            
        # Remove the document itself
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        if doc_id in self.documents:
            del self.documents[doc_id]
        
        # Remove from BM25 retriever
        self.bm25_retriever.remove_document(doc_id)
            
        logger.info(f"Removed document: {doc_id}")
            
    def search(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.6,
        rerank: bool = True,
        rerank_strategy: str = "hybrid",
        use_hybrid: bool = True,
        hybrid_ratio: float = 0.7  # Weight for semantic search vs BM25
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using semantic similarity.
        
        Args:
            query: Search query
            k: Maximum number of results to return
            threshold: Minimum similarity score required
            rerank: Whether to apply re-ranking
            rerank_strategy: Re-ranking strategy to use
            use_hybrid: Whether to use hybrid search (combine semantic and BM25)
            hybrid_ratio: Weight for semantic search (1-hybrid_ratio for BM25)
            
        Returns:
            List of documents with similarity scores above the threshold
        """
        if not self.embeddings:
            logger.warning("No embeddings available for search")
            return []
            
        try:
            # Perform BM25 search for sparse retrieval
            bm25_results = self.bm25_retriever.search(query, k=k*2)
            
            # For hybrid search we'll combine results
            if use_hybrid:
                semantic_results = self._semantic_search(query, k=k*2, threshold=threshold)
                combined_results = self._combine_search_results(
                    semantic_results, 
                    bm25_results, 
                    hybrid_ratio=hybrid_ratio
                )
                results = combined_results
            else:
                # Semantic search only if not using hybrid
                results = self._semantic_search(query, k=k*2, threshold=threshold)
                
            # Apply re-ranking if enabled
            if rerank and results:
                results = self.reranker.rerank_results(query, results, strategy=rerank_strategy)
            
            # Final limiting to k results
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
            
    def _semantic_search(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        try:
            # Generate embedding for the query
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            query_embedding = np.array(result['embedding'])
            
            # Calculate cosine similarity between query and all documents
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                similarity = self._calculate_similarity(query_embedding, doc_embedding)
                if similarity >= threshold:
                    similarities.append((doc_id, similarity))
                
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Create result objects
            results = []
            for doc_id, score in similarities[:k]:
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    result = {
                        'id': doc_id,
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': float(score),
                        'semantic_score': float(score),  # Keep original semantic score
                        'retrieval_method': 'semantic'
                    }
                    
                    # Check if this is a chunk and add parent info
                    if self.chunker.is_chunk_id(doc_id):
                        parent_id = self.chunker.get_parent_id(doc_id)
                        if parent_id and parent_id in self.documents:
                            result['parent_id'] = parent_id
                            # Add parent document title or filename if available
                            parent_metadata = self.documents[parent_id].get('metadata', {})
                            if 'title' in parent_metadata:
                                result['parent_title'] = parent_metadata['title']
                            elif 'filename' in parent_metadata:
                                result['parent_title'] = parent_metadata['filename']
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}")
            return []
    
    def _combine_search_results(
        self, 
        semantic_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]],
        hybrid_ratio: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic and BM25 search results.
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            hybrid_ratio: Weight for semantic search (1-hybrid_ratio for BM25)
            
        Returns:
            Combined and normalized results
        """
        # Create a mapping of document IDs to results
        combined_map = {}
        
        # Process semantic results
        for result in semantic_results:
            doc_id = result['id']
            combined_map[doc_id] = result
            
        # Process BM25 results and merge with semantic
        for result in bm25_results:
            doc_id = result['id']
            
            if doc_id in combined_map:
                # Document exists in both - combine scores
                sem_score = combined_map[doc_id].get('semantic_score', 0)
                bm25_score = result.get('bm25_score', 0)
                
                # Normalize BM25 score (they can be >1)
                max_bm25 = max([r.get('bm25_score', 0) for r in bm25_results])
                if max_bm25 > 0:
                    norm_bm25 = bm25_score / max_bm25
                else:
                    norm_bm25 = 0
                
                # Weighted combination
                combined_score = (sem_score * hybrid_ratio) + (norm_bm25 * (1 - hybrid_ratio))
                
                combined_map[doc_id]['score'] = combined_score
                combined_map[doc_id]['bm25_score'] = bm25_score
                combined_map[doc_id]['retrieval_method'] = 'hybrid'
            else:
                # Document only in BM25 results
                # First ensure we have the content and metadata
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    result['content'] = doc['content']
                    result['metadata'] = doc['metadata']
                    
                    # Check if this is a chunk and add parent info
                    if self.chunker.is_chunk_id(doc_id):
                        parent_id = self.chunker.get_parent_id(doc_id)
                        if parent_id and parent_id in self.documents:
                            result['parent_id'] = parent_id
                            # Add parent document title or filename if available
                            parent_metadata = self.documents[parent_id].get('metadata', {})
                            if 'title' in parent_metadata:
                                result['parent_title'] = parent_metadata['title']
                            elif 'filename' in parent_metadata:
                                result['parent_title'] = parent_metadata['filename']
                    
                    # Normalize BM25 score
                    max_bm25 = max([r.get('bm25_score', 0) for r in bm25_results])
                    if max_bm25 > 0:
                        norm_bm25 = result.get('bm25_score', 0) / max_bm25
                    else:
                        norm_bm25 = 0
                    
                    # Weight the normalized BM25 score
                    result['score'] = norm_bm25 * (1 - hybrid_ratio)
                    combined_map[doc_id] = result
        
        # Convert map back to list and sort by combined score
        combined_results = list(combined_map.values())
        combined_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return combined_results
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure embeddings are 1D arrays
        if vec1.ndim > 1:
            vec1 = vec1.flatten()
        if vec2.ndim > 1:
            vec2 = vec2.flatten()
            
        # Check for zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
        
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
            logger.info(f"Saved {len(self.embeddings)} embeddings to {embeddings_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            
        # Save document info
        docs_file = save_dir / 'documents.json'
        try:
            with open(docs_file, 'w') as f:
                json.dump(self.documents, f)
            logger.info(f"Saved {len(self.documents)} documents to {docs_file}")
        except Exception as e:
            logger.error(f"Error saving documents metadata: {str(e)}")
            
        # Save parent-chunks mapping
        chunks_file = save_dir / 'parent_chunks.json'
        try:
            with open(chunks_file, 'w') as f:
                json.dump(self.parent_chunks, f)
            logger.info(f"Saved parent-chunks mapping to {chunks_file}")
        except Exception as e:
            logger.error(f"Error saving parent-chunks mapping: {str(e)}")
            
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
                logger.info(f"Loaded {len(self.embeddings)} embeddings from {embeddings_file}")
            except Exception as e:
                logger.error(f"Error loading embeddings: {str(e)}")
                
        # Load document info
        docs_file = load_dir / 'documents.json'
        if docs_file.exists():
            try:
                with open(docs_file, 'r') as f:
                    self.documents = json.load(f)
                    
                # Rebuild the BM25 index from loaded documents
                for doc_id, doc_info in self.documents.items():
                    content = doc_info.get('content', '')
                    if content:
                        self.bm25_retriever.add_document(doc_id, content)
                        
                logger.info(f"Loaded {len(self.documents)} documents from {docs_file}")
            except Exception as e:
                logger.error(f"Error loading documents metadata: {str(e)}")
                
        # Load parent-chunks mapping
        chunks_file = load_dir / 'parent_chunks.json'
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r') as f:
                    self.parent_chunks = json.load(f)
                logger.info(f"Loaded parent-chunks mapping from {chunks_file}")
            except Exception as e:
                logger.error(f"Error loading parent-chunks mapping: {str(e)}") 