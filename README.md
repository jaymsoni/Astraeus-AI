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
- **Configurable AI Models**: Supports using different chat and embedding models (Google Gemini, Ollama)

## Technology Stack

- **Backend**: Flask, Python
- **AI/ML**: 
  - LangChain for AI orchestration
  - LangGraph for agent workflows
  - OpenAI/Google Gemini for language models
  - Sentence Transformers for embeddings
- **Database**: ChromaDB for vector storage
- **Frontend**: React

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
   - Add your API keys and configure settings. **Ensure you set `GOOGLE_API_KEY` if using Google Gemini models and `OLLAMA_BASE_URL` if using Ollama models.**

5. Run the application:
   ```bash
   python backend/app.py
   ```

## Model Configuration

Astraeus supports using different AI models for chat and embeddings. You can configure the default models using environment variables and switch between available models in the UI.

-   **`GOOGLE_API_KEY`**: Your API key for accessing Google Gemini models. Required if you want to use any of the `gemini` models.
-   **`OLLAMA_BASE_URL`**: The base URL for your Ollama instance (e.g., `http://localhost:11434`). Required if you want to use any of the `ollama/` models. Ensure your Ollama server is running and the necessary models are pulled.
-   **`LLM_MODEL`**: (Optional) The default chat model to use when the application starts. Defaults to `gemini-2.0-flash`. Can be overridden by the UI selection.
-   **`EMBEDDING_MODEL`**: (Optional) The default embedding model to use when the application starts. Defaults to `models/embedding-001`. Can be overridden by the UI selection.

The UI provides dropdowns to select the desired chat and embedding models from the list of `AVAILABLE_MODELS` defined in the backend (`backend/app.py`). Your selection in the UI will override the default models set by environment variables for the current session.

## API Endpoints

### `GET /`
- Health check endpoint
- Returns: Status message

### `POST /upload`
- Upload and process a document
- Accepts: Multipart form data with file and optional metadata
- Returns: Processing results

### `POST /query`
- Search through processed documents and get a response from the selected LLM.
- Accepts: JSON with query string, optional `conversation_id`, and `model_id` (though the backend now primarily uses the globally selected model).
- Returns: Relevant document matches and the LLM's answer.

### `GET /api/models`
- Get the list of available chat and embedding models.
- Returns: JSON containing available models grouped by type.

### `POST /api/models`
- Update the currently selected chat or embedding model.
- Accepts: JSON with `model_type` ('chat' or 'embedding') and `model_id`.
- Returns: Success message.

## Project Structure

```
astraeus/
├── backend/              # Backend services
│   ├── agents/          # AI agents
│   ├── utils/           # Utility functions
│   └── models/          # Data models
│   └── config.py        # Configuration settings
├── data/                # Data storage
│   ├── uploads/         # Uploaded files
│   └── vector_store/    # Vector embeddings
├── frontend/            # Frontend application
│   ├── public/          # Static assets (index.html, css, js)
│   └── src/             # React source files
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