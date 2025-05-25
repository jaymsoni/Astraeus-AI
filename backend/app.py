from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import json
from werkzeug.utils import secure_filename
import google.generativeai as genai
import uuid
import re
from typing import Optional, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests

# Import configuration
from backend.config import (
    UPLOAD_FOLDER, MAX_CONTENT_LENGTH, DEBUG, 
    GOOGLE_API_KEY, LLM_MODEL as DEFAULT_LLM_MODEL, MAX_CONCURRENT_UPLOADS,
    LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    get_config_dict
)

# Import agents and utils
from backend.agents.data_intake import DataIntakeAgent
from backend.agents.learning import LearningAgent
from backend.utils.file_handlers import (
    allowed_file, save_file, validate_file, load_document,
    get_file_info, cleanup_file, FileProcessingError
)
from backend.utils.conversation import ConversationManager

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure GenAI if API key is available
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google Generative AI configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI: {e}")
else:
    logger.warning("GOOGLE_API_KEY not found in environment. LLM generation will not work.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure app using settings from config.py
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['DEBUG'] = DEBUG

# Initialize agents
data_intake_agent = DataIntakeAgent()
learning_agent = LearningAgent(data_intake_agent.embedding_manager)

# Initialize conversation manager
conversation_manager = ConversationManager()

# Initialize thread pool for file processing
file_processor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_UPLOADS)

# Initialize file system watcher
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"File modified: {event.src_path}")
            # Handle file modification if needed

    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            # Handle file deletion if needed

observer = Observer()
observer.schedule(FileChangeHandler(), UPLOAD_FOLDER, recursive=False)
observer.start()

# Learning-related command patterns
LEARNING_COMMANDS = {
    'analyze': r'analyze (?:document|file) (.+)',
    'study_guide': r'(?:create|generate) (?:a )?(?:study guide|learning guide) (?:for|from) (.+)',
    'quick_guide': r'(?:create|generate) (?:a )?quick (?:study|learning) guide (?:for|from) (.+)',
    'detailed_guide': r'(?:create|generate) (?:a )?detailed (?:study|learning) guide (?:for|from) (.+)'
}

# Document mention pattern
DOCUMENT_MENTION_PATTERN = r'@([\w\s.-]+)'

# Model configuration
AVAILABLE_MODELS = {
    'chat': [
        {'id': 'gemini-2.0-flash', 'name': 'Gemini 2.0 Flash'},
        {'id': 'gemini-1.5-flash', 'name': 'Gemini 1.5 Flash'},
        {'id': 'gemini-1.0-pro', 'name': 'Gemini 1.0 Pro'},
        {'id': 'ollama/llama2', 'name': 'Llama 2 (Ollama)'},
        {'id': 'ollama/mistral', 'name': 'Mistral (Ollama)'},
        {'id': 'ollama/codellama', 'name': 'CodeLlama (Ollama)'},
        {'id': 'ollama/neural-chat', 'name': 'Neural Chat (Ollama)'}
    ],
    'embedding': [
        {'id': 'models/embedding-001', 'name': 'Embedding 001'},
        {'id': 'models/embedding-002', 'name': 'Embedding 002'},
        {'id': 'ollama/nomic-embed-text', 'name': 'Nomic Embed Text (Ollama)'}
    ]
}

# Add Ollama client configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

# Global variable to store selected models
# Initialize with default values from config
_selected_models = {
    'chat': DEFAULT_LLM_MODEL,
    'embedding': os.getenv("EMBEDDING_MODEL", "models/embedding-001") # Assuming EMBEDDING_MODEL env var or default
}

def get_llm_response(prompt: str, model_id: str) -> str:
    """
    Get response from either Google Gemini or Ollama model.
    
    Args:
        prompt: The prompt to send to the model
        model_id: The ID of the model to use
        
    Returns:
        The model's response text
    """
    logger.info(f"Attempting to get LLM response using model_id: {model_id}")
    try:
        if model_id.startswith('ollama/'):
            # Use Ollama API
            model_name = model_id.replace('ollama/', '')
            logger.info(f"Using Ollama model: {model_name}")
            response = requests.post(
                f'{OLLAMA_BASE_URL}/api/generate',
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False
                }
            )
            response.raise_for_status()
            logger.info("Successfully received response from Ollama.")
            return response.json()['response']
        else:
            # Use Google Gemini API
            logger.info(f"Using Google Gemini model: {model_id}")
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not configured")
            model = genai.GenerativeModel(model_id)
            response = model.generate_content(prompt)
            logger.info("Successfully received response from Google Gemini.")
            return response.text
    except Exception as e:
        logger.error(f"Error getting LLM response for model {model_id}: {str(e)}")
        raise

@app.route('/')
def home():
    """Home/health check endpoint."""
    return jsonify({
        'status': 'success',
        'message': 'Astraeus AI Assistant API is running'
    })

@app.route('/config')
def get_config():
    """Return the current configuration (for debugging only)."""
    # Only show this in debug mode for security
    if app.config['DEBUG']:
        return jsonify({
            'status': 'success',
            'config': get_config_dict()
        })
    return jsonify({
        'status': 'error',
        'message': 'Configuration endpoint is only available in debug mode'
    }), 403

@app.route('/upload', methods=['POST'])
async def upload_file():
    """
    Handle file uploads and process them using the Data Intake Agent.
    Expects a file in the request and optional metadata as JSON.
    """
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file provided'
        }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400

    try:
        # Validate file
        is_valid, error_message = validate_file(file)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_message
            }), 400

        # Get metadata if provided
        try:
            metadata = json.loads(request.form.get('metadata', '{}'))
        except json.JSONDecodeError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid metadata JSON'
            }), 400

        # Save the file and get the filename and path
        filename, file_path = save_file(file)

        # Get file info for metadata
        file_info = get_file_info(file_path)
        metadata.update(file_info)

        # Process the file using the Data Intake Agent
        try:
            success = await asyncio.get_event_loop().run_in_executor(
                file_processor,
                data_intake_agent.process_file,
                file_path,
                metadata
            )

            if not success:
                # Clean up the saved file if processing failed
                cleanup_file(file_path)
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to process file'
                }), 500

            return jsonify({
                'status': 'success',
                'message': 'File processed successfully',
                'data': {
                    'filename': filename,
                    'metadata': metadata
                }
            })

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            cleanup_file(file_path)
            return jsonify({
                'status': 'error',
                'message': f'Error processing file: {str(e)}'
            }), 500

    except FileProcessingError as e:
        logger.error(f"File processing error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred'
        }), 500

@app.route('/documents/suggestions', methods=['GET'])
def get_document_suggestions():
    """
    Get list of available documents for mention suggestions.
    """
    try:
        # Get all documents from the data intake agent
        documents = data_intake_agent.get_all_documents()
        
        # Format suggestions
        suggestions = []
        for doc_id, doc_info in documents.items():
            # Skip document chunks
            if '#chunk-' in doc_id:
                continue
                
            metadata = doc_info.get('metadata', {})
            suggestions.append({
                'id': doc_id,
                'name': metadata.get('filename', doc_id),
                'type': metadata.get('type', 'unknown'),
                'summary': metadata.get('summary', ''),
                'metadata': metadata
            })
        
        return jsonify({
            'status': 'success',
            'suggestions': suggestions
        })
        
    except Exception as e:
        logger.error(f"Error getting document suggestions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def process_document_mentions(query: str) -> str:
    """
    Process document mentions in the query.
    
    Args:
        query: The user's query string
        
    Returns:
        Processed query with document mentions replaced with document IDs
    """
    try:
        # Get all available documents
        documents = data_intake_agent.get_all_documents()
        
        # Create a mapping of document names to IDs
        doc_map = {}
        for doc_id, doc_info in documents.items():
            if '#chunk-' in doc_id:
                continue
            metadata = doc_info.get('metadata', {})
            name = metadata.get('filename', doc_id)
            doc_map[name.lower()] = doc_id
        
        # Replace mentions with document IDs
        def replace_mention(match):
            mention = match.group(1).strip().lower()
            # Find the best matching document
            for name, doc_id in doc_map.items():
                if mention in name.lower() or name.lower() in mention:
                    return f"@{doc_id}"
            return match.group(0)  # Return original if no match found
            
        processed_query = re.sub(DOCUMENT_MENTION_PATTERN, replace_mention, query)
        return processed_query
        
    except Exception as e:
        logger.error(f"Error processing document mentions: {str(e)}")
        return query

@app.route('/query', methods=['POST'])
def query_documents():
    """
    Handle document queries and learning-related requests.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No query provided'
        }), 400
    
    user_query = data['query']
    conversation_id = data.get('conversation_id', str(uuid.uuid4()))
    
    try:
        # Process document mentions in the query
        processed_query = process_document_mentions(user_query)
        
        # Check for learning-related commands
        learning_response = handle_learning_commands(processed_query)
        if learning_response:
            return jsonify(learning_response)
            
        # If not a learning command, proceed with normal document search
        # Extract search parameters with defaults
        k = data.get('num_results', 3)
        threshold = data.get('threshold', 0.6)
        rerank = data.get('rerank', True)
        rerank_strategy = data.get('rerank_strategy', 'hybrid')
        
        # New parameters for hybrid search
        use_hybrid = data.get('use_hybrid', True)
        hybrid_ratio = data.get('hybrid_ratio', 0.7)
        
        # Parameter to control whether to send source documents to frontend
        show_sources = data.get('show_sources', False)
        
        # Get conversation history
        conversation_history = conversation_manager.get_conversation_history(conversation_id)
        
        # Add user's message to history
        conversation_manager.add_message(conversation_id, "user", processed_query)
        
        # 1. Retrieve relevant documents with enhanced search
        logger.info(f"Retrieving documents for query: '{processed_query}' with threshold {threshold}")
        search_results = data_intake_agent.search_documents(
            query=processed_query,
            k=k,
            threshold=threshold,
            rerank=rerank,
            rerank_strategy=rerank_strategy,
            use_hybrid=use_hybrid,
            hybrid_ratio=hybrid_ratio
        )
        
        # 2. Prepare context
        context_parts = []
        
        # Add conversation history if available
        if conversation_history:
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in conversation_history
            ])
            context_parts.append(f"Previous conversation:\n{history_text}")
        
        # Add document context if available
        if search_results:
            # Format documents by origin/source
            grouped_docs = {}
            for res in search_results:
                doc_id = res.get('id', '')
                source = ''
                
                # Handle chunked documents
                if 'parent_id' in res:
                    parent_id = res['parent_id']
                    if 'parent_title' in res:
                        source = res['parent_title']
                    else:
                        source = parent_id
                else:
                    source = doc_id
                
                # Add snippet if available, otherwise full content
                content = res.get('snippet', res.get('content', ''))
                if source not in grouped_docs:
                    grouped_docs[source] = []
                
                # Add score info and retrieval method
                score_info = res.get('score', 0)
                retrieval_method = res.get('retrieval_method', 'unknown')
                scored_content = f"{content}\n[Relevance score: {score_info:.2f}, Method: {retrieval_method}]"
                
                grouped_docs[source].append(scored_content)
            
            # Format the document content
            docs_text = ""
            for source, contents in grouped_docs.items():
                docs_text += f"Document: {source}\n"
                for i, content in enumerate(contents, 1):
                    if len(contents) > 1:
                        docs_text += f"Excerpt {i}:\n{content}\n\n"
                    else:
                        docs_text += f"Content:\n{content}\n\n"
            
            context_parts.append(f"Relevant documents:\n{docs_text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Limit context length
        max_context_length = 8000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "... [context truncated]"
            
        # 3. Create assistant-like prompt
        prompt = f"""You are Astraeus, a helpful and knowledgeable AI assistant. You MUST format ALL responses using Markdown.

REQUIRED FORMATTING RULES:
1. Start with a clear heading using # or ##
2. Use **bold** for important points and emphasis
3. Use bullet points (*) or numbered lists (1.) for structured information
4. Use `code` for technical terms, file names, and technical concepts
5. Use > for quoting or referencing document content
6. Break your response into sections using ## subheadings
7. Use emojis at the start of sections for better readability
8. Use proper spacing between sections for readability

Example format:
# Analysis Title 📊
## Key Points 💡
* **First point**: Description
* **Second point**: Description

## Details 📝
> Relevant quote from document

## Summary ✨
Final thoughts and recommendations

---

Your responses should be:
- Professional yet friendly
- Clear and well-structured
- Based on the provided context and documents
- Honest about what you don't know

IMPORTANT: When you cite or reference information from the documents, include a brief citation at the end of the relevant sentence or paragraph, like "[Source: Document Name]".

{context if context else "Note: No relevant documents or conversation history found."}

User's question: {processed_query}

Remember to use proper markdown formatting in your response:"""
        
        # Use the globally selected chat model
        current_llm_model = _selected_models.get('chat', DEFAULT_LLM_MODEL)
        logger.info(f"Generating response using model: {current_llm_model}")
        
        try:
            # Get response from the appropriate model
            generated_answer = get_llm_response(prompt, current_llm_model)
            logger.info("LLM response received.")
            
            # Add assistant's response to conversation history
            conversation_manager.add_message(conversation_id, "assistant", generated_answer)

        except Exception as llm_error:
            logger.error(f"Error calling LLM API: {llm_error}")
            return jsonify({'status': 'error', 'message': 'Failed to generate answer from LLM.'}), 500
            
        # 4. Return results, conditionally including search results
        response_data = {
            'status': 'success',
            'conversation_id': conversation_id,
            'answer': generated_answer,
            'source_count': len(search_results) if search_results else 0,
            'search_params': {
                'threshold': threshold,
                'rerank': rerank,
                'rerank_strategy': rerank_strategy,
                'use_hybrid': use_hybrid,
                'hybrid_ratio': hybrid_ratio
            }
        }
        
        # Only include search results if explicitly requested
        if show_sources:
            response_data['results'] = search_results
            
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in /query endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred during processing.'
        }), 500

def handle_learning_commands(query: str) -> Optional[Dict]:
    """
    Handle learning-related commands in user queries.
    
    Args:
        query: The user's query string
        
    Returns:
        Dict containing the response if it's a learning command, None otherwise
    """
    try:
        # Check for analyze command
        analyze_match = re.match(LEARNING_COMMANDS['analyze'], query, re.IGNORECASE)
        if analyze_match:
            doc_id = analyze_match.group(1).strip()
            result = learning_agent.analyze_document(doc_id)
            return format_learning_response(result, 'analysis')
            
        # Check for study guide commands
        guide_match = re.match(LEARNING_COMMANDS['study_guide'], query, re.IGNORECASE)
        if guide_match:
            doc_id = guide_match.group(1).strip()
            result = learning_agent.generate_study_guide(doc_id, style='comprehensive')
            return format_learning_response(result, 'study_guide')
            
        # Check for quick guide command
        quick_match = re.match(LEARNING_COMMANDS['quick_guide'], query, re.IGNORECASE)
        if quick_match:
            doc_id = quick_match.group(1).strip()
            result = learning_agent.generate_study_guide(doc_id, style='quick')
            return format_learning_response(result, 'study_guide')
            
        # Check for detailed guide command
        detailed_match = re.match(LEARNING_COMMANDS['detailed_guide'], query, re.IGNORECASE)
        if detailed_match:
            doc_id = detailed_match.group(1).strip()
            result = learning_agent.generate_study_guide(doc_id, style='detailed')
            return format_learning_response(result, 'study_guide')
            
        return None
        
    except Exception as e:
        logger.error(f"Error handling learning command: {str(e)}")
        return {
            'status': 'error',
            'message': f"Error processing learning request: {str(e)}"
        }

def format_learning_response(result: Dict, response_type: str) -> Dict:
    """
    Format the learning agent's response for the chat interface.
    
    Args:
        result: The raw result from the learning agent
        response_type: Type of response ('analysis' or 'study_guide')
        
    Returns:
        Formatted response for the chat interface
    """
    if result['status'] != 'success':
        return result
        
    if response_type == 'analysis':
        analysis = result['analysis']
        return {
            'status': 'success',
            'type': 'analysis',
            'content': {
                'overview': f"Analysis of document {result['doc_id']}:",
                'key_concepts': analysis['key_concepts'],
                'learning_objectives': analysis['learning_objectives'],
                'difficulty': analysis['difficulty_level'],
                'prerequisites': analysis['prerequisite_concepts']
            }
        }
    else:  # study_guide
        guide = result['content']
        return {
            'status': 'success',
            'type': 'study_guide',
            'style': result['style'],
            'content': guide
        }

@app.route('/documents/<string:doc_id>', methods=['DELETE'])
def delete_document_route(doc_id: str):
    """
    Handle deleting a document by its ID.
    """
    try:
        logger.info(f"Received DELETE request for doc_id: {doc_id}")
        success = data_intake_agent.delete_document(doc_id)
        if success:
            logger.info(f"Successfully processed deletion for {doc_id}")
            return jsonify({
                'status': 'success',
                'message': f'Document {doc_id} deleted successfully.'
            })
        else:
            logger.warning(f"Deletion failed for {doc_id} in agent")
            # The agent likely logged the specific error
            return jsonify({
                'status': 'error',
                'message': f'Failed to delete document {doc_id}. Check server logs.'
            }), 500

    except Exception as e:
        logger.error(f"Error in /documents/<doc_id> DELETE endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while attempting to delete the document.'
        }), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """
    Retrieve a list of all processed documents with their metadata and summaries.
    """
    try:
        # Get all documents from the data intake agent
        documents = data_intake_agent.get_all_documents()
        
        # Format the response to include only necessary information
        formatted_documents = []
        for doc_id, doc_info in documents.items():
            # Skip document chunks
            if '#chunk-' in doc_id:
                continue
                
            # Extract metadata
            metadata = doc_info.get('metadata', {})
            
            # Create formatted document object
            formatted_doc = {
                'id': doc_id,
                'filename': metadata.get('filename', doc_id),
                'size': metadata.get('size', 0),
                'type': metadata.get('type', 'unknown'),
                'uploaded_at': metadata.get('uploaded_at', ''),
                'summary': metadata.get('summary', 'No summary available')
            }
            
            formatted_documents.append(formatted_doc)
        
        return jsonify({
            'status': 'success',
            'documents': formatted_documents
        })
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to retrieve documents: {str(e)}"
        }), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models."""
    return jsonify({
        'status': 'success',
        'models': AVAILABLE_MODELS
    })

@app.route('/api/models', methods=['POST'])
def update_model():
    """
    Update the selected model.
    """
    data = request.get_json()
    if not data or 'model_type' not in data or 'model_id' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required fields'
        }), 400
        
    model_type = data['model_type']
    model_id = data['model_id']
    
    # Validate model type and ID
    if model_type not in AVAILABLE_MODELS:
        return jsonify({
            'status': 'error',
            'message': f'Invalid model type: {model_type}'
        }), 400
        
    if not any(m['id'] == model_id for m in AVAILABLE_MODELS[model_type]):
        return jsonify({
            'status': 'error',
            'message': f'Invalid model ID: {model_id}'
        }), 400
    
    try:
        # Update the global selected models
        _selected_models[model_type] = model_id
            
        logger.info(f"Updated selected {model_type} model to {model_id}")

        return jsonify({
            'status': 'success',
            'message': f'Successfully updated {model_type} model to {model_id}',
            'selected_models': _selected_models # Optional: send back current selection
        })
        
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to update model: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'status': 'error',
        'message': f'File is too large. Maximum size is {MAX_CONTENT_LENGTH/1024/1024}MB'
    }), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# The __main__ block is not needed when running via run.py
# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(host='0.0.0.0', port=5000, debug=True) 