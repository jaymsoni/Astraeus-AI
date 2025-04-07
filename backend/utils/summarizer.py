"""
Document summarization utility for Astraeus.
Provides functionality to generate concise summaries of documents using the Gemini API.
"""
import logging
from typing import Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    logger.warning("GOOGLE_API_KEY environment variable is not set. Summarization will not work.")
else:
    genai.configure(api_key=api_key)

# Define the model to use for summarization - use the correct model name
SUMMARY_MODEL = "gemini-1.5-pro"  # Updated model name

def generate_document_summary(content: str, max_length: int = 200) -> Optional[str]:
    """
    Generate a concise summary of document content using the Gemini API.
    
    Args:
        content: The document content to summarize
        max_length: Maximum length of the summary in words
        
    Returns:
        A summary of the document or None if summarization fails
    """
    if not api_key:
        logger.error("Cannot generate summary: GOOGLE_API_KEY not set")
        return None
        
    if not content or len(content.strip()) < 50:
        logger.warning("Content too short to summarize")
        return content
    
    try:
        # Prepare the prompt for summarization
        prompt = f"""Create a clear and concise summary of the following document. 
The summary should:
- Be at most {max_length} words
- Capture the main ideas and key points
- Be written in a professional tone
- Highlight the document's purpose and central conclusions
- Be easily understood by someone unfamiliar with the document

Document content:
{content[:8000]}  # Limit content to avoid token limits

Summary:"""

        # Call the Gemini API
        model = genai.GenerativeModel(SUMMARY_MODEL)
        response = model.generate_content(prompt)
        
        # Extract and process the summary
        summary = response.text.strip()
        
        logger.info(f"Successfully generated summary ({len(summary.split())} words)")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating document summary: {str(e)}")
        
        # Try with a fallback model if the first attempt fails
        try:
            logger.info("Trying with fallback model gemini-pro")
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            summary = response.text.strip()
            logger.info(f"Successfully generated summary with fallback model ({len(summary.split())} words)")
            return summary
        except Exception as fallback_error:
            logger.error(f"Fallback summarization attempt also failed: {str(fallback_error)}")
            return None 