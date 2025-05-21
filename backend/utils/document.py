"""Document processing utilities for Astraeus."""
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document:
    """
    A standardized representation of a document in the system.
    
    This class encapsulates all document-related data and metadata,
    providing a consistent interface for document processing.
    """
    
    def __init__(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None
    ):
        """
        Initialize a new Document instance.
        
        Args:
            doc_id: Unique identifier for the document
            content: The document's text content
            metadata: Optional metadata about the document
            file_path: Optional path to the source file
        """
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.file_path = file_path
        
        # Add basic metadata if not provided
        if not self.metadata.get('created_at'):
            self.metadata['created_at'] = datetime.now().isoformat()
        if not self.metadata.get('doc_id'):
            self.metadata['doc_id'] = doc_id
        if file_path and not self.metadata.get('file_path'):
            self.metadata['file_path'] = file_path
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary representation."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata,
            'file_path': self.file_path
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a Document instance from a dictionary."""
        return cls(
            doc_id=data['doc_id'],
            content=data['content'],
            metadata=data.get('metadata', {}),
            file_path=data.get('file_path')
        )

class DocumentProcessor(ABC):
    """
    Abstract base class for document processors.
    
    This interface defines the standard methods for processing different types of documents.
    Implementations should handle specific file formats and processing requirements.
    """
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if the processor can handle this file type
        """
        pass
        
    @abstractmethod
    def process(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Process a document and return a Document instance.
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata about the document
            
        Returns:
            Document: A processed Document instance
            
        Raises:
            ValueError: If the file cannot be processed
        """
        pass
        
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        pass

class TextDocumentProcessor(DocumentProcessor):
    """Processor for plain text documents."""
    
    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.txt', '.md', '.text'))
        
    def process(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            doc_id = os.path.basename(file_path)
            extracted_metadata = self.extract_metadata(file_path)
            
            if metadata:
                extracted_metadata.update(metadata)
                
            return Document(
                doc_id=doc_id,
                content=content,
                metadata=extracted_metadata,
                file_path=file_path
            )
        except Exception as e:
            logger.error(f"Error processing text document {file_path}: {str(e)}")
            raise ValueError(f"Failed to process text document: {str(e)}")
            
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            file_info = os.stat(file_path)
            return {
                'file_size': file_info.st_size,
                'modified_at': datetime.fromtimestamp(file_info.st_mtime).isoformat(),
                'created_at': datetime.fromtimestamp(file_info.st_ctime).isoformat(),
                'file_type': 'text'
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}

class PDFDocumentProcessor(DocumentProcessor):
    """Processor for PDF documents."""
    
    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
        
    def process(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
                
            doc_id = os.path.basename(file_path)
            extracted_metadata = self.extract_metadata(file_path)
            
            if metadata:
                extracted_metadata.update(metadata)
                
            return Document(
                doc_id=doc_id,
                content=content.strip(),
                metadata=extracted_metadata,
                file_path=file_path
            )
        except Exception as e:
            logger.error(f"Error processing PDF document {file_path}: {str(e)}")
            raise ValueError(f"Failed to process PDF document: {str(e)}")
            
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            file_info = os.stat(file_path)
            
            return {
                'file_size': file_info.st_size,
                'modified_at': datetime.fromtimestamp(file_info.st_mtime).isoformat(),
                'created_at': datetime.fromtimestamp(file_info.st_ctime).isoformat(),
                'file_type': 'pdf',
                'page_count': len(reader.pages),
                'author': reader.metadata.author if reader.metadata else None,
                'title': reader.metadata.title if reader.metadata else None
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF {file_path}: {str(e)}")
            return {}

class DocumentProcessorFactory:
    """
    Factory class for creating appropriate document processors.
    
    This class manages the registration and instantiation of document processors
    for different file types.
    """
    
    def __init__(self):
        self._processors = []
        self._register_default_processors()
        
    def _register_default_processors(self):
        """Register the default set of document processors."""
        self.register_processor(TextDocumentProcessor())
        self.register_processor(PDFDocumentProcessor())
        # Add more default processors here
        
    def register_processor(self, processor: DocumentProcessor):
        """
        Register a new document processor.
        
        Args:
            processor: The DocumentProcessor instance to register
        """
        self._processors.append(processor)
        
    def get_processor(self, file_path: str) -> Optional[DocumentProcessor]:
        """
        Get the appropriate processor for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Optional[DocumentProcessor]: The appropriate processor, or None if none found
        """
        for processor in self._processors:
            if processor.can_process(file_path):
                return processor
        return None
        
    def process_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Process a document using the appropriate processor.
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata about the document
            
        Returns:
            Document: The processed document
            
        Raises:
            ValueError: If no suitable processor is found
        """
        processor = self.get_processor(file_path)
        if not processor:
            raise ValueError(f"No processor found for file: {file_path}")
        return processor.process(file_path, metadata) 