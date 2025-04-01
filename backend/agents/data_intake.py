from typing import List, Dict, Any
from backend.utils.file_handlers import load_document, allowed_file

class DataIntakeAgent:
    def __init__(self):
        self.documents = {}  # Store processed documents

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
                return False
                
            # Load and process the document
            content = load_document(file_path)
            
            if not content:
                return False
                
            # Store the processed content with metadata
            doc_id = file_path.split('/')[-1]
            self.documents[doc_id] = {
                'content': content,
                'metadata': metadata or {}
            }
            
            return True
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
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