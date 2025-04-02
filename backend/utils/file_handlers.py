"""File handling utilities for Astraeus."""
import os
import logging
from typing import Optional, Tuple, Dict, Any
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import csv
import json
import xml.etree.ElementTree as ET
import yaml

from backend.config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename: str) -> bool:
    """
    Check if a file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file) -> Tuple[str, str]:
    """
    Save an uploaded file securely.
    
    Args:
        file: File object from request.files
        
    Returns:
        Tuple[str, str]: Tuple containing (filename, filepath)
    """
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    logger.info(f"Saved file: {filename} to {file_path}")
    return filename, file_path

def load_document(file_path: str) -> Optional[str]:
    """
    Load and read the contents of a document.
    Supports multiple file formats including PDF, TXT, CSV, JSON, etc.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        Optional[str]: Contents of the file if successful, None otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        file_extension = file_path.split('.')[-1].lower()
        
        # Text files
        if file_extension in ['txt', 'md', 'text']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # PDF files
        elif file_extension == 'pdf':
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        
        # CSV files
        elif file_extension in ['csv']:
            text = []
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader, None)
                if headers:
                    text.append(', '.join(headers))
                for row in csv_reader:
                    text.append(', '.join(row))
            return '\n'.join(text)
        
        # JSON files
        elif file_extension == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Pretty-print JSON to make it more readable
                return json.dumps(data, indent=2)
        
        # XML files
        elif file_extension == 'xml':
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            def process_element(element, indent=0):
                result = ' ' * indent + element.tag
                if element.text and element.text.strip():
                    result += ': ' + element.text.strip()
                result += '\n'
                
                for child in element:
                    result += process_element(child, indent + 2)
                return result
            
            return process_element(root)
        
        # YAML files
        elif file_extension in ['yaml', 'yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return yaml.dump(data, sort_keys=False, default_flow_style=False)
        
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            # Try as plain text as a fallback
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                logger.error(f"Could not read {file_path} as text")
                return f"File type {file_extension} not fully supported yet. Please upload a different format."
            
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return None

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get metadata about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict[str, Any]: Dictionary containing file metadata
    """
    try:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_extension = filename.split('.')[-1].lower()
        mod_time = os.path.getmtime(file_path)
        
        return {
            'filename': filename,
            'extension': file_extension,
            'size': file_size,
            'modified': mod_time,
            'path': file_path
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        return {
            'filename': os.path.basename(file_path),
            'error': str(e)
        } 