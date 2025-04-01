from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json

# Import agents and utils using absolute paths from the project root
from backend.agents.data_intake import DataIntakeAgent
from backend.utils.file_handlers import allowed_file # Removed save_file, assuming it's not used here yet
# from backend.config import UPLOAD_FOLDER # Assuming config.py exists in backend

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads' # Temporary placeholder, replace with config import later
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize agents
data_intake_agent = DataIntakeAgent()

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Astraeus AI Assistant API is running'
    })

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

        # Save the file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the file using the Data Intake Agent
        success = data_intake_agent.process_file(
            file_path=file_path,
            metadata=metadata
        )

        if not success:
             # Clean up the saved file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
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
        # Clean up the saved file in case of any exception during processing
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Placeholder for query endpoint - needs implementation
@app.route('/query', methods=['POST'])
def query_documents():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'status': 'error', 'message': 'No query provided'}), 400

    # TODO: Implement query logic using an appropriate agent (e.g., QueryAgent)
    # results = query_agent.search(data['query'])
    results = [{'id': 'doc1', 'content': 'Placeholder content', 'score': 0.9}]

    return jsonify({'status': 'success', 'results': results})


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'status': 'error',
        'message': 'File is too large. Maximum size is 16MB'
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