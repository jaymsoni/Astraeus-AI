"""File handling utilities for Astraeus."""
import os
from typing import Optional

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv', 'json'}

def allowed_file(filename: str) -> bool:
    """
    Check if a file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_document(file_path: str) -> Optional[str]:
    """
    Load and read the contents of a document.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        Optional[str]: Contents of the file if successful, None otherwise
    """
    try:
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None 