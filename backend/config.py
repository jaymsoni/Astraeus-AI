"""
Configuration management for Astraeus.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# File storage settings
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(DATA_DIR, "uploads"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", os.path.join(DATA_DIR, "vector_store"))

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-pro")

# File handling settings
# Tuple of allowed file extensions for upload
ALLOWED_EXTENSIONS = {
    # Text files
    "txt", "md", "text",
    
    # Document files 
    "pdf", "doc", "docx", "rtf", "odt",
    
    # Spreadsheets
    "csv", "xls", "xlsx", "ods",
    
    # Presentations
    "ppt", "pptx", "odp",
    
    # Data files
    "json", "xml", "yaml", "yml",
}

# Maximum upload file size in bytes (16MB)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# Application settings
DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
ENV = os.getenv("FLASK_ENV", "production")

# Function to get configuration as a dictionary (useful for debugging)
def get_config_dict():
    """Return all configuration values as a dictionary."""
    return {key: value for key, value in globals().items() 
            if key.isupper() and not key.startswith('_')} 