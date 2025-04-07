#!/usr/bin/env python3
"""
Script to reset the Astraeus document database.
Deletes all uploaded files and clears the vector database.
"""
import os
import shutil
import json
import logging
from pathlib import Path

from backend.config import UPLOAD_FOLDER, VECTOR_STORE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_directory(directory_path):
    """Clear all files in the given directory."""
    if not os.path.exists(directory_path):
        logger.warning(f"Directory {directory_path} does not exist")
        return

    logger.info(f"Clearing directory: {directory_path}")
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
                logger.info(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Deleted directory: {item_path}")
        except Exception as e:
            logger.error(f"Error deleting {item_path}: {e}")

def reset_vector_store():
    """Reset the vector store by creating empty files."""
    # Ensure the directory exists
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    
    # Create empty JSON files
    empty_files = {
        'embeddings.json': {},
        'documents.json': {},
        'parent_chunks.json': {}
    }
    
    for filename, empty_content in empty_files.items():
        file_path = os.path.join(VECTOR_STORE_PATH, filename)
        try:
            with open(file_path, 'w') as f:
                json.dump(empty_content, f)
            logger.info(f"Reset {filename} to empty state")
        except Exception as e:
            logger.error(f"Error resetting {filename}: {e}")

def main():
    """Main function to reset the database."""
    try:
        # Clear uploaded files
        logger.info("Starting database reset...")
        
        # 1. Delete all uploaded files
        logger.info("Clearing uploaded files...")
        clear_directory(UPLOAD_FOLDER)
        
        # 2. Reset vector store
        logger.info("Resetting vector store...")
        reset_vector_store()
        
        logger.info("Database reset completed successfully!")
        print("\nAll documents have been deleted and the vector database has been reset.")
        print(f"- Cleared uploads directory: {UPLOAD_FOLDER}")
        print(f"- Reset vector store: {VECTOR_STORE_PATH}")
        
    except Exception as e:
        logger.error(f"Error during database reset: {e}")
        print(f"\nError: Failed to reset database: {e}")

if __name__ == "__main__":
    main() 