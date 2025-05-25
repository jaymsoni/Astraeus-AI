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
TEMP_FOLDER = os.path.join(DATA_DIR, "temp")

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # 120 seconds default timeout

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
    
    # Images
    "jpg", "jpeg", "png", "gif", "bmp",
}

# File size limits
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16MB default
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 16 * 1024 * 1024))  # 16MB default
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024 * 1024))  # 1MB default

# File processing settings
ENCODING_DETECTION_SAMPLE_SIZE = int(os.getenv("ENCODING_DETECTION_SAMPLE_SIZE", 4096))
FILE_HASH_CHUNK_SIZE = int(os.getenv("FILE_HASH_CHUNK_SIZE", 4096))
MAX_CONCURRENT_UPLOADS = int(os.getenv("MAX_CONCURRENT_UPLOADS", 5))

# Document processing settings
PDF_PAGE_LIMIT = int(os.getenv("PDF_PAGE_LIMIT", 1000))
EXCEL_ROW_LIMIT = int(os.getenv("EXCEL_ROW_LIMIT", 10000))
IMAGE_SIZE_LIMIT = int(os.getenv("IMAGE_SIZE_LIMIT", 4096))  # Max dimension in pixels

# Application settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ENV = os.getenv("FLASK_ENV", "production")

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = os.getenv("LOG_FILE", os.path.join(DATA_DIR, "logs", "app.log"))

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Function to get configuration as a dictionary (useful for debugging)
def get_config_dict():
    """Return a dictionary of all configuration settings."""
    return {
        "UPLOAD_FOLDER": UPLOAD_FOLDER,
        "VECTOR_STORE_PATH": VECTOR_STORE_PATH,
        "TEMP_FOLDER": TEMP_FOLDER,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "LLM_MODEL": LLM_MODEL,
        "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
        "OLLAMA_TIMEOUT": OLLAMA_TIMEOUT,
        "MAX_CONTENT_LENGTH": MAX_CONTENT_LENGTH,
        "MAX_FILE_SIZE": MAX_FILE_SIZE,
        "CHUNK_SIZE": CHUNK_SIZE,
        "MAX_CONCURRENT_UPLOADS": MAX_CONCURRENT_UPLOADS,
        "DEBUG": DEBUG
    } 