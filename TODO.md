# Astraeus Project TODO List

This file tracks planned features, improvements, and bug fixes.

## High Priority

- [x] ~~**Refine Semantic Search Relevance:**~~
    - ✅ ~~Experiment with different similarity thresholds (currently default is 0.6).~~
    - ✅ ~~Implement document chunking for more granular embeddings and search.~~
    - ✅ ~~Potentially try different embedding models.~~
    - ✅ ~~Consider adding a re-ranking step.~~

- [x] ~~**RAG Enhancements (2025 Standard):**~~
    - ✅ ~~Advanced document chunking with multiple strategies (recursive, sentence, paragraph)~~
    - ✅ ~~Hybrid search combining dense (embedding) and sparse (BM25) retrieval~~
    - ✅ ~~Weighted combination of retrieval methods~~
    - ✅ ~~Enhanced result display with retrieval method indicators~~

- [x] ~~**Document Management (API & UI):**~~
    - ✅ ~~Implement API endpoint and frontend UI for deleting documents.~~
    - ✅ ~~Clear associated embeddings when a document is deleted.~~

## Medium Priority

- [ ] **Configuration Management:**
    - [x] ~~Create a dedicated `backend/config.py` file.~~
    - [ ] Move settings like `UPLOAD_FOLDER`, `VECTOR_STORE_PATH`, and `EMBEDDING_MODEL` from `.env` or hardcoded values into `config.py`.

- [x] ~~**UI Enhancements (Phase 1):**~~
    - ✅ ~~Add loading indicators during file upload and query processing.~~
    - ✅ ~~Add better visual feedback for errors.~~
    - ✅ ~~Implement markdown support for better message formatting.~~
    - ✅ ~~Improve display of search results (e.g., show snippets, highlight matches).~~

- [ ] **Document Processing Structure:**
    - [ ] Create a dedicated Document class to standardize document representation
    - [ ] Implement a DocumentProcessor interface for different file types
    - [ ] Add metadata extraction pipeline for better context
    - [ ] Build document repository pattern for consistent storage access
    - [ ] Refine chunk storage/retrieval with advanced indexing

- [ ] **Improved Chat Response Flow:**
    - [x] ~~Hide retrieved documents from user responses~~
    - [x] ~~Pass documents as context to LLM only~~
    - [x] ~~Add optional toggle for users to view source documents if needed~~
    - [x] ~~Improve citation format when referencing documents~~
    - [ ] Add confidence indicators for LLM responses

- [x] ~~**Document Summarization:**~~
    - ✅ ~~Implement document summary generation using Gemini API~~
    - ✅ ~~Store summaries as document metadata~~
    - ✅ ~~Add frontend UI to display document summaries~~
    - ✅ ~~Allow users to toggle summary visibility~~

## New Features

- [ ] **File System Access Capabilities:**
    - [ ] Implement secure file system access for reading local files
    - [ ] Add file path parsing and validation in user queries
    - [ ] Create a file content analyzer for various file types
    - [ ] Implement context-aware responses based on file content
    - [ ] Add support for common file formats (txt, csv, json, etc.)
    - [ ] Implement file content summarization and analysis
    - [ ] Add file access permissions and security measures
    - [ ] Create a file indexing system for quick access
    - [ ] Implement file change detection and updates
    - [ ] Add support for structured data analysis (e.g., task lists, schedules)

## Low Priority / Future Ideas

- [ ] **Expand File Type Support:** Add reliable text extraction for `.docx`, `.xlsx`, etc. in `file_handlers.py`.
- [ ] **Implement Specialized Agents:**
    - Finance Agent
    - Learning Agent
    - Travel Agent
- [ ] **User Authentication/Accounts:** If needed for multi-user scenarios.
- [ ] **Testing:** Add unit and integration tests.

## Completed Features ✅

1. **Document Management**:
   - API endpoint for document deletion
   - Frontend UI for document management
   - Delete confirmation dialog
   - Proper cleanup of associated embeddings

2. **Configuration**:
   - Basic config.py structure
   - Environment variable support

3. **UI/UX Improvements**:
   - Loading indicators
   - Error feedback system
   - Markdown support in chat
   - File upload progress indicators
   - Enhanced search results display
   
4. **Search Enhancements**:
   - Document chunking for better granularity
   - Configurable similarity thresholds
   - Re-ranking with keyword and LLM approaches
   - Hybrid search strategy
   - Relevant snippet extraction
   - Context grouping by document source
   
5. **RAG 2025 Upgrades**:
   - Advanced recursive chunking strategies
   - Natural language aware document splitting
   - BM25 sparse retrieval implementation
   - Hybrid dense-sparse retrieval with configurable weights
   - Improved document metadata and tracking
   - Visual distinction of retrieval methods in UI 

6. **Document Summarization**:
   - Automatic document summary generation
   - Gemini API integration for summary creation
   - Summary display in document list
   - Toggle UI for showing/hiding summaries 