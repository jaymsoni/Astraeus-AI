# Astraeus - Personal AI Assistant

Astraeus is an intelligent personal assistant that helps manage and analyze various types of documents, including bills, learning materials, travel documents, and more. It uses advanced AI capabilities to process and understand documents, providing smart insights and assistance.

## Features

- **Multi-Modal Document Processing**: Handle various file types (PDFs, images, text files, etc.)
- **Intelligent Document Analysis**: Extract and understand information from documents using AI
- **Vector-Based Search**: Efficiently search through processed documents
- **Specialized Agents**: 
  - Data Intake Agent: Process and store documents
  - Finance Agent: Analyze bills and financial documents
  - Learning Agent: Track and assist with learning goals
  - Travel Agent: Manage travel documents and itineraries

## Technology Stack

- **Backend**: Flask, Python
- **AI/ML**: 
  - LangChain for AI orchestration
  - LangGraph for agent workflows
  - OpenAI/Google Gemini for language models
  - Sentence Transformers for embeddings
- **Database**: ChromaDB for vector storage
- **Frontend**: (Coming soon)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd astraeus
   ```

2. Create a virtual environment:
   ```bash
   conda create -n astraeus_env python=3.9
   conda activate astraeus_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys and configure settings

5. Run the application:
   ```bash
   python backend/app.py
   ```

## API Endpoints

### `GET /`
- Health check endpoint
- Returns: Status message

### `POST /upload`
- Upload and process a document
- Accepts: Multipart form data with file and optional metadata
- Returns: Processing results

### `POST /query`
- Search through processed documents
- Accepts: JSON with query string
- Returns: Relevant document matches

## Project Structure

```
astraeus/
├── backend/              # Backend services
│   ├── agents/          # AI agents
│   ├── utils/           # Utility functions
│   └── models/          # Data models
├── data/                # Data storage
│   ├── uploads/         # Uploaded files
│   └── vector_store/    # Vector embeddings
├── frontend/            # Frontend application
└── requirements.txt     # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 