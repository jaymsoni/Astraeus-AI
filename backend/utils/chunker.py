"""
Document chunking utilities for better semantic search.

This module provides tools to split documents into smaller chunks
to improve embedding quality and search relevance.
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import re
import logging
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Enumeration of different chunking strategies."""
    SIMPLE = "simple"           # Basic character-based chunking
    RECURSIVE = "recursive"     # Recursive splitting based on multiple separators
    SENTENCE = "sentence"       # Sentence-aware splitting
    PARAGRAPH = "paragraph"     # Paragraph-based splitting
    HYBRID = "hybrid"           # Combines multiple strategies

class DocumentChunker:
    """
    Handles splitting documents into smaller, more manageable chunks.
    
    This improves semantic search by:
    1. Creating more focused embeddings
    2. Allowing more precise retrieval of relevant content
    3. Handling longer documents more effectively
    """
    
    def __init__(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
        strategy: Union[str, ChunkingStrategy] = ChunkingStrategy.HYBRID,
        separators: List[str] = None
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy to use (simple, recursive, sentence, paragraph, hybrid)
            separators: Optional list of separators for recursive splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Convert string strategy to enum if needed
        if isinstance(strategy, str):
            try:
                self.strategy = ChunkingStrategy(strategy)
            except ValueError:
                logger.warning(f"Unknown strategy: {strategy}, falling back to HYBRID")
                self.strategy = ChunkingStrategy.HYBRID
        else:
            self.strategy = strategy
        
        # Default separators for recursive splitting in order of precedence
        self.separators = separators or [
            "\n\n",     # Double line break (paragraphs)
            "\n",       # Single line break
            ". ",       # End of sentence
            "! ",       # End of sentence (exclamation)
            "? ",       # End of sentence (question)
            ";",        # Semicolon
            ":",        # Colon
            ",",        # Comma
            " ",        # Space (word boundary)
            ""          # Character (last resort)
        ]
        
    def chunk_document(
        self, 
        doc_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split document into chunks with metadata.
        
        Args:
            doc_id: Original document identifier
            content: Full text content to chunk
            metadata: Optional metadata to associate with each chunk
            
        Returns:
            List of chunk dictionaries with ID, content, and metadata
        """
        if not content:
            logger.warning(f"Empty content for document: {doc_id}")
            return []
            
        # Create chunks based on selected strategy
        if self.strategy == ChunkingStrategy.SIMPLE:
            chunks = self._split_text_simple(content)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._split_text_recursive(content)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._split_text_sentences(content)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._split_text_paragraphs(content)
        else:  # Default to HYBRID
            chunks = self._split_text_hybrid(content)
        
        # Base metadata (copied from original document)
        base_metadata = metadata.copy() if metadata else {}
        
        # Create chunk objects with IDs and metadata
        result = []
        for i, chunk_text in enumerate(chunks):
            # Create unique chunk ID from document ID
            chunk_id = f"{doc_id}#chunk-{i+1}"
            
            # Add chunk-specific metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "parent_doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_chunk": True,
                "strategy": self.strategy.value,
                "char_length": len(chunk_text),
                "word_count": len(chunk_text.split()),
            })
            
            # Add chunk to results
            result.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": chunk_metadata
            })
            
        logger.info(f"Split document {doc_id} into {len(result)} chunks using {self.strategy.value} strategy")
        return result
    
    def _split_text_simple(self, text: str) -> List[str]:
        """
        Simple splitting strategy based on character count.
        
        Args:
            text: Text content to split
            
        Returns:
            List of text chunks
        """
        # Handle edge cases
        if not text:
            return []
            
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Define the potential end of this chunk
            end = min(start + self.chunk_size, len(text))
            
            # If we're not at the end of the text, try to find a natural breakpoint
            if end < len(text):
                # Look for a space within the overlap zone
                last_space_pos = text.rfind(" ", start, end)
                
                # If found within a reasonable distance, use it
                if last_space_pos != -1 and last_space_pos > start + (self.chunk_size // 2):
                    end = last_space_pos + 1  # Include the space
            
            # Extract the chunk and add it
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move to next position, accounting for overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end - self.chunk_overlap)
        
        return chunks
    
    def _split_text_recursive(self, text: str) -> List[str]:
        """
        Recursive splitting strategy using multiple separators.
        
        Args:
            text: Text content to split
            
        Returns:
            List of text chunks
        """
        # Handle edge cases
        if not text:
            return []
            
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to split by each separator in order
        chunks = self._recursive_split(text, 0)
        return chunks
        
    def _recursive_split(self, text: str, separator_idx: int = 0) -> List[str]:
        """Helper for recursive splitting using the given separator index."""
        # Base case: we've tried all separators or text is small enough
        if separator_idx >= len(self.separators) or len(text) <= self.chunk_size:
            return [text]
            
        separator = self.separators[separator_idx]
        
        # If text doesn't contain this separator, try the next one
        if separator and separator not in text:
            return self._recursive_split(text, separator_idx + 1)
            
        # Try splitting with current separator
        split_text = text.split(separator)
        
        # If this splitting produces appropriate-sized chunks, use it
        final_chunks = []
        current_chunk = []
        current_length = 0
        
        for segment in split_text:
            segment_with_sep = segment + (separator if separator else "")
            segment_len = len(segment_with_sep)
            
            # If adding this segment exceeds chunk size, finalize the current chunk
            if current_length + segment_len > self.chunk_size and current_length > 0:
                final_chunks.append(separator.join(current_chunk) + (separator if separator else ""))
                current_chunk = []
                current_length = 0
                
            # If segment itself is too large, recursively split it with next separator
            if segment_len > self.chunk_size:
                subsegments = self._recursive_split(segment_with_sep, separator_idx + 1)
                
                # If we have a current chunk, finalize it before adding subsegments
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk) + (separator if separator else ""))
                    current_chunk = []
                    current_length = 0
                
                final_chunks.extend(subsegments)
            else:
                # Add segment to current chunk
                current_chunk.append(segment)
                current_length += segment_len
        
        # Don't forget the last chunk
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))
            
        # Ensure we have proper overlap between chunks
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._add_chunk_overlap(final_chunks)
            
        return final_chunks
    
    def _split_text_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks that respect sentence boundaries.
        
        Args:
            text: Text content to split
            
        Returns:
            List of text chunks
        """
        # Simple sentence boundary detection
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_length + sentence_len > self.chunk_size and current_length > 0:
                chunks.append(' '.join(current_chunk))
                
                # Calculate overlap
                overlap_sentences = []
                overlap_length = 0
                
                # Add sentences from the end of the previous chunk for overlap
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1  # +1 for space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def _split_text_paragraphs(self, text: str) -> List[str]:
        """
        Split text into chunks that respect paragraph boundaries.
        
        Args:
            text: Text content to split
            
        Returns:
            List of text chunks
        """
        # Split by double newlines (common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_len = len(paragraph)
            
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if current_length + paragraph_len > self.chunk_size and current_length > 0:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If paragraph itself is too large, recursively split it
            if paragraph_len > self.chunk_size:
                # Use sentence splitting for large paragraphs
                paragraph_chunks = self._split_text_sentences(paragraph)
                
                # If we have a current chunk, finalize it before adding paragraph chunks
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                chunks.extend(paragraph_chunks)
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_length += paragraph_len + 2  # +2 for "\n\n"
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks
    
    def _split_text_hybrid(self, text: str) -> List[str]:
        """
        Hybrid approach that combines multiple strategies.
        
        Args:
            text: Text content to split
            
        Returns:
            List of text chunks
        """
        # First try paragraph-based splitting
        chunks = self._split_text_paragraphs(text)
        
        # If this results in chunks that are too large, refine them
        refined_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                # Use recursive splitting for large chunks
                refined_chunks.extend(self._split_text_recursive(chunk))
            else:
                refined_chunks.append(chunk)
                
        return refined_chunks
    
    def _add_chunk_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if not chunks or len(chunks) <= 1:
            return chunks
            
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Try to find overlap from the end of previous chunk
            if len(prev_chunk) >= self.chunk_overlap:
                overlap_text = prev_chunk[-self.chunk_overlap:]
                
                # If current chunk doesn't already start with the overlap
                if not current_chunk.startswith(overlap_text):
                    current_chunk = overlap_text + current_chunk
            
            result.append(current_chunk)
            
        return result
    
    def is_chunk_id(self, doc_id: str) -> bool:
        """
        Check if an ID represents a chunk.
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if this is a chunk ID, False otherwise
        """
        return "#chunk-" in doc_id
        
    def get_parent_id(self, chunk_id: str) -> Optional[str]:
        """
        Extract the parent document ID from a chunk ID.
        
        Args:
            chunk_id: The chunk ID to parse
            
        Returns:
            Parent document ID or None if not a valid chunk ID
        """
        if not self.is_chunk_id(chunk_id):
            return None
            
        # Extract the part before "#chunk-"
        parts = chunk_id.split("#chunk-")
        if len(parts) >= 2:
            return parts[0]
            
        return None 