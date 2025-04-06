from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import json
from werkzeug.utils import secure_filename
import google.generativeai as genai
import uuid

# Import configuration
from backend.config import (
    UPLOAD_FOLDER, MAX_CONTENT_LENGTH, DEBUG, 
    GOOGLE_API_KEY, LLM_MODEL,
    get_config_dict
)

# Import agents and utils
from backend.agents.data_intake import DataIntakeAgent
from backend.utils.file_handlers import allowed_file, save_file
from backend.utils.conversation import ConversationManager

# Set up logging
logging.basicConfig(level=logging.INFO)
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

# Initialize conversation manager
conversation_manager = ConversationManager()

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
def upload_file():
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

    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'File type not allowed'
        }), 400

    try:
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Get metadata if provided
        metadata = json.loads(request.form.get('metadata', '{}'))

        # Save the file and get the filename and path
        filename, file_path = save_file(file)

        # Process the file using the Data Intake Agent
        success = data_intake_agent.process_file(
            file_path=file_path,
            metadata=metadata
        )

        if not success:
             # Clean up the saved file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.warning(f"Removed file due to processing failure: {file_path}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to process file'
            }), 500

        return jsonify({
            'status': 'success',
            'message': 'File processed successfully',
            'data': {'filename': filename}
        })

    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        # Clean up the saved file in case of any exception during processing
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/query', methods=['POST'])
def query_documents():
    """
    Search through processed documents and generate an answer using an LLM.
    Expects a JSON body with 'query', optional 'conversation_id', and optional parameters.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No query provided'
        }), 400
    
    user_query = data['query']
    conversation_id = data.get('conversation_id', str(uuid.uuid4()))
    
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
    
    try:
        # Get conversation history
        conversation_history = conversation_manager.get_conversation_history(conversation_id)
        
        # Add user's message to history
        conversation_manager.add_message(conversation_id, "user", user_query)
        
        # 1. Retrieve relevant documents with enhanced search
        logger.info(f"Retrieving documents for query: '{user_query}' with threshold {threshold}")
        search_results = data_intake_agent.search_documents(
            query=user_query,
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

User's question: {user_query}

Remember to use proper markdown formatting in your response:"""
        
        logger.info(f"Generating response using model: {LLM_MODEL}")
        
        if not GOOGLE_API_KEY:
             return jsonify({'status': 'error', 'message': 'LLM generation is disabled. GOOGLE_API_KEY not configured.'}), 500

        try:
            # Use the specified LLM model
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(prompt)
            generated_answer = response.text
            logger.info("LLM response received.")
            
            # Add assistant's response to conversation history
            conversation_manager.add_message(conversation_id, "assistant", generated_answer)

        except Exception as llm_error:
            logger.error(f"Error calling Gemini API: {llm_error}")
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
            'message': 'An error occurred during the query process.'
        }), 500

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