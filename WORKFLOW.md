# Astraeus Project Workflow

This document outlines the complete workflow for setting up, running, and maintaining the Astraeus AI Assistant project.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Development Workflow](#development-workflow)
4. [Document Management](#document-management)
5. [System Maintenance](#system-maintenance)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

## Project Overview

Astraeus is an AI-powered document assistant that uses RAG (Retrieval-Augmented Generation) techniques to help users interact with their documents. Key features include:

- Document uploading, management, and deletion
- Semantic search with hybrid retrieval methods
- Automatic document summarization
- Natural language query processing
- Interactive chat interface

The system uses the Gemini API for generating summaries and responses, and employs multiple search techniques to retrieve relevant information.

## Setup and Installation

### Prerequisites

- Python 3.8+ 
- pip (Python package manager)
- Node.js and npm (only if modifying frontend)
- API keys for Gemini (Google AI)

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd astraeus
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv astraeus_env
   source astraeus_env/bin/activate  # On Windows: astraeus_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root with the following:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   FLASK_DEBUG=1  # Set to 0 for production
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   python run.py
   ```

2. **Access the application**
   
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Development Workflow

### Directory Structure

```
astraeus/
├── backend/
│   ├── agents/
│   │   ├── data_intake.py
│   │   └── ...
│   ├── utils/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── file_handlers.py
│   │   ├── reranker.py
│   │   ├── summarizer.py
│   │   └── ...
│   ├── app.py
│   └── config.py
├── data/
│   ├── uploads/      # Stores uploaded documents
│   └── vector_store/ # Stores embeddings and document metadata
├── frontend/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── index.html
├── .env
├── requirements.txt
├── reset_database.py
├── reset.sh
└── run.py
```

### Making Changes

#### Backend Changes

1. **Update code**: Make changes to files in the `backend/` directory
2. **Test**: Restart the Flask server to test changes
3. **Debug**: Use Flask's debug mode for hot reloading

#### Frontend Changes

1. **Modify HTML/CSS/JS**: Edit files in the `frontend/` directory
2. **Refresh**: Simply refresh the browser to see changes

#### API Endpoint Workflow

When creating a new API endpoint:

1. Add the route in `backend/app.py`
2. Implement the required functionality in appropriate agent or utility files
3. Test using tools like Postman or cURL
4. Update frontend to use the new endpoint

## Document Management

### Upload Workflow

1. User uploads document through web interface
2. Backend validates file type
3. File is saved to `data/uploads/` directory
4. Document content is extracted based on file type
5. Content is processed and summarized using the Gemini API
6. Document is chunked for efficient retrieval
7. Embeddings are generated for the document and its chunks
8. Document metadata and embeddings are stored in `data/vector_store/`

### Search and Retrieval Workflow

1. User submits a query in the chat interface
2. Query is processed to generate embeddings
3. System performs hybrid search:
   - Semantic search using embeddings
   - Sparse search using BM25 algorithm
4. Results are combined and reranked based on relevance
5. Top results are passed to the LLM as context
6. LLM generates a response using RAG technique
7. Response is formatted with markdown and displayed to user

### Document Panel Interaction

The document panel provides a centralized interface for:
- Viewing all uploaded documents
- Reading document summaries
- Quickly asking questions about specific documents
- Deleting unwanted documents

### Document Deletion

Documents can be deleted through:
1. The main document list
2. The document management panel
3. The API endpoint directly

When a document is deleted:
1. The file is removed from `data/uploads/`
2. The document and its chunks are removed from memory
3. Associated embeddings are removed from the vector store
4. The vector store is saved to disk

## System Maintenance

### Clearing All Documents

To reset the system and remove all documents:

1. **Using the reset script** (recommended)
   ```bash
   ./reset.sh
   ```
   This will prompt for confirmation before deleting all documents.

2. **Directly using Python**
   ```bash
   python reset_database.py
   ```
   This performs the reset without confirmation.

The reset process:
1. Deletes all files in the `data/uploads/` directory
2. Resets the vector store files to empty states:
   - `embeddings.json`
   - `documents.json`
   - `parent_chunks.json`

### Backup and Restore

#### Backup
To back up your data, copy the following directories:
```bash
cp -r data/uploads /backup/location/uploads
cp -r data/vector_store /backup/location/vector_store
```

#### Restore
To restore from backup:
```bash
cp -r /backup/location/uploads data/uploads
cp -r /backup/location/vector_store data/vector_store
```

### Updating API Keys

If you need to update your API keys:
1. Edit the `.env` file
2. Restart the application

## API Reference

### Document Management Endpoints

- `GET /documents`: Retrieve list of all documents with metadata and summaries
- `POST /upload`: Upload a new document
- `DELETE /documents/<doc_id>`: Delete a specific document

### Query Endpoints

- `POST /query`: Process a natural language query

### Configuration Endpoints

- `GET /config`: Get current configuration (debug mode only)

## Troubleshooting

### Common Issues

#### Server Won't Start
- Check if the port is already in use
- Verify that all required packages are installed
- Ensure API keys are configured correctly

#### Upload Errors
- Verify file is of an allowed type
- Check file size is under the maximum limit (default: 16MB)
- Ensure upload directory is writable

#### Search Returns No Results
- Check that documents have been properly uploaded
- Verify that embeddings were generated correctly
- Try adjusting the similarity threshold (default: 0.6)

#### Document Processing Errors
- Check API key validity for Gemini models
- Ensure network connectivity for API requests
- Check storage space for vector database

### Logs

The application logs detailed information about its operations:
- Backend logs are printed to the console
- Important events include file processing, embedding generation, and search operations

### Getting Help

If you encounter issues not covered here:
1. Check the detailed logs
2. Examine the error messages
3. Review relevant code in the corresponding module 