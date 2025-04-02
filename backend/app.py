from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import json
from werkzeug.utils import secure_filename
import google.generativeai as genai

# Import configuration
from backend.config import (
    UPLOAD_FOLDER, MAX_CONTENT_LENGTH, DEBUG, 
    GOOGLE_API_KEY, LLM_MODEL,
    get_config_dict
)

# Import agents and utils
from backend.agents.data_intake import DataIntakeAgent
from backend.utils.file_handlers import allowed_file, save_file

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
    Expects a JSON body with 'query' and optional 'num_results', 'threshold'.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No query provided'
        }), 400
    
    user_query = data['query']
    k = data.get('num_results', 3) # Limit context slightly
    threshold = data.get('threshold', 0.6)
    
    try:
        # 1. Retrieve relevant documents
        logger.info(f"Retrieving documents for query: '{user_query}'")
        search_results = data_intake_agent.search_documents(
            query=user_query,
            k=k,
            threshold=threshold
        )
        
        if not search_results:
            logger.info("No relevant documents found above threshold.")
            # Optionally, still try asking the LLM without context, or return directly
            # For now, return specific message
            return jsonify({
                'status': 'success',
                'message': 'No relevant documents found to answer the query.',
                'results': [], # Keep results structure consistent
                'answer': "I couldn't find any relevant information in your documents to answer that specific question."
            })
        
        # 2. Augment: Prepare context for LLM
        context = "\n\n---\n\n".join([f"Document: {res['id']}\nContent: {res['content']}" for res in search_results])
        
        # Limit context length (simple approach, more sophisticated needed for production)
        max_context_length = 8000 # Adjust based on model limits
        if len(context) > max_context_length:
            context = context[:max_context_length] + "... [context truncated]"
            
        # 3. Generate: Create prompt and call LLM
        prompt = f"""Based *only* on the following context from the documents provided, answer the user's question. If the context doesn't contain the answer, say so explicitly. Do not use any prior knowledge.

Context:
{context}

Question: {user_query}

Answer:"""
        
        logger.info(f"Generating LLM response using model: {LLM_MODEL}")
        
        if not GOOGLE_API_KEY:
             return jsonify({'status': 'error', 'message': 'LLM generation is disabled. GOOGLE_API_KEY not configured.'}), 500

        try:
            # Use the specified LLM model from config
            model = genai.GenerativeModel(LLM_MODEL) 
            response = model.generate_content(prompt)
            generated_answer = response.text
            logger.info("LLM response received.")

        except Exception as llm_error:
            logger.error(f"Error calling Gemini API: {llm_error}")
            return jsonify({'status': 'error', 'message': 'Failed to generate answer from LLM.'}), 500
            
        # 4. Return combined results
        return jsonify({
            'status': 'success',
            'results': search_results, # Return the source documents too
            'answer': generated_answer
        })
    
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