from typing import List, Dict, Any, Optional
import os
import logging

from backend.utils.file_handlers import load_document, allowed_file, save_file, get_file_info
from backend.utils.embeddings import EmbeddingManager
from backend.config import UPLOAD_FOLDER, VECTOR_STORE_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIntakeAgent:
    def __init__(self, use_chunks: bool = True, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the data intake agent.
        
        Args:
            use_chunks: Whether to use document chunking
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between document chunks
        """
        self.documents = {}  # Store processed documents
        # Initialize embedding manager with chunking options
        self.embedding_manager = EmbeddingManager(
            use_chunks=use_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Load existing embeddings if they exist
        if os.path.exists(VECTOR_STORE_PATH):
            logger.info(f"Loading existing embeddings from {VECTOR_STORE_PATH}")
            self.embedding_manager.load_embeddings(VECTOR_STORE_PATH)
        else:
            logger.info(f"No existing embeddings found at {VECTOR_STORE_PATH}")

    def process_file(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Process an uploaded file and store its contents.
        
        Args:
            file_path: Path to the uploaded file
            metadata: Optional metadata about the file
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            if not allowed_file(file_path):
                logger.warning(f"File type not allowed: {file_path}")
                return False
                
            # Get file info
            file_info = get_file_info(file_path)
            if metadata is None:
                metadata = {}
            
            # Merge file info with provided metadata
            metadata.update(file_info)
            
            # Load and process the document
            logger.info(f"Processing file: {file_path}")
            content = load_document(file_path)
            
            if not content:
                logger.error(f"Failed to extract content from file: {file_path}")
                return False
                
            # Store the processed content with metadata
            doc_id = os.path.basename(file_path)
            self.documents[doc_id] = {
                'content': content,
                'metadata': metadata or {}
            }
            
            # Add document to embedding manager
            logger.info(f"Generating embeddings for: {doc_id}")
            self.embedding_manager.add_document(doc_id, content, metadata)
            
            # Save embeddings to disk
            logger.info(f"Saving embeddings to {VECTOR_STORE_PATH}")
            self.embedding_manager.save_embeddings(VECTOR_STORE_PATH)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return False
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a processed document by its ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dict containing document content and metadata
        """
        return self.documents.get(doc_id, {})
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all processed documents.
        
        Returns:
            List of dictionaries containing document content and metadata
        """
        return [
            {'id': doc_id, **doc_data}
            for doc_id, doc_data in self.documents.items()
        ]

    def search_documents(
        self, 
        query: str, 
        k: int = 4, 
        threshold: float = 0.6,
        rerank: bool = True,
        rerank_strategy: str = "hybrid",
        use_hybrid: bool = True,
        hybrid_ratio: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search through processed documents using semantic similarity.
        
        Args:
            query: The search query string
            k: Maximum number of results to return 
            threshold: Minimum similarity score required
            rerank: Whether to apply re-ranking
            rerank_strategy: The re-ranking strategy to use
            use_hybrid: Whether to use hybrid search (combine semantic and BM25)
            hybrid_ratio: Weight for semantic search vs BM25 (0-1)
            
        Returns:
            List of matching documents with relevance scores
        """
        logger.info(f"Searching for: '{query}' with threshold {threshold}")
        
        results = self.embedding_manager.search(
            query=query, 
            k=k, 
            threshold=threshold,
            rerank=rerank,
            rerank_strategy=rerank_strategy,
            use_hybrid=use_hybrid,
            hybrid_ratio=hybrid_ratio
        )
        
        # Post-process results for better display
        processed_results = []
        for result in results:
            # Deep copy to avoid modifying original
            processed = result.copy()
            
            # Add snippet context around the query terms if available
            if 'content' in processed:
                processed['snippet'] = self._extract_relevant_snippet(
                    processed['content'], 
                    query, 
                    max_length=300
                )
            
            processed_results.append(processed)
            
        logger.info(f"Found {len(processed_results)} results for query: '{query}'")
        return processed_results
        
    def _extract_relevant_snippet(self, content: str, query: str, max_length: int = 300) -> str:
        """
        Extract a relevant snippet from content based on the query.
        
        Args:
            content: The full document content
            query: The search query
            max_length: Maximum length of the snippet
            
        Returns:
            The most relevant text snippet
        """
        if not content or not query:
            return ""
            
        # Convert to lowercase for case-insensitive matching
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Remove stop words from query for better matching
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'with'}
        query_terms = [term for term in query_lower.split() if term not in stop_words and len(term) > 2]
        
        # Find the best starting position based on term matches
        best_pos = 0
        best_score = 0
        
        # Step through the content looking for matches
        pos = 0
        while pos < len(content_lower):
            current_score = sum(1 for term in query_terms if term in content_lower[pos:pos+max_length])
            if current_score > best_score:
                best_score = current_score
                best_pos = pos
            pos += max_length // 4  # Smaller step size for better coverage
            
        # Extract the snippet
        end_pos = min(best_pos + max_length, len(content))
        snippet = content[best_pos:end_pos]
        
        # Add ellipsis if needed
        if best_pos > 0:
            snippet = "..." + snippet
        if end_pos < len(content):
            snippet = snippet + "..."
            
        return snippet

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.
        Removes the document file, its entry, and its embedding.

        Args:
            doc_id: The ID (filename) of the document to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            # Construct the expected file path
            file_path = os.path.join(UPLOAD_FOLDER, doc_id)

            # 1. Remove from EmbeddingManager (memory)
            logger.info(f"Removing document from embedding manager: {doc_id}")
            self.embedding_manager.remove_document(doc_id)

            # 2. Remove from internal documents dictionary (memory)
            if doc_id in self.documents:
                logger.info(f"Removing document from internal storage: {doc_id}")
                del self.documents[doc_id]

            # 3. Delete the physical file
            if os.path.exists(file_path):
                logger.info(f"Deleting file: {file_path}")
                os.remove(file_path)
            else:
                # It's okay if the file doesn't exist, maybe it was already deleted
                logger.warning(f"File not found, skipping deletion: {file_path}")

            # 4. Persist the changes by saving the updated embeddings state
            logger.info(f"Saving updated embeddings after removing {doc_id}")
            self.embedding_manager.save_embeddings(VECTOR_STORE_PATH)

            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False 