# Astraeus Project TODO List

This file tracks planned features, improvements, and bug fixes.

## High Priority

- [ ] **Refine Semantic Search Relevance:**
    - Experiment with different similarity thresholds (currently default is 0.6).
    - Implement document chunking for more granular embeddings and search.
    - Potentially try different embedding models.
    - Consider adding a re-ranking step.

## Medium Priority

- [ ] **Document Management:**
    - Implement API endpoint and frontend UI for deleting documents.
    - Clear associated embeddings when a document is deleted.
- [ ] **Configuration Management:**
    - Create a dedicated `backend/config.py` file.
    - Move settings like `UPLOAD_FOLDER`, `VECTOR_STORE_PATH`, `EMBEDDING_MODEL` from `.env` or hardcoded values into `config.py`.
- [ ] **UI Enhancements:**
    - Add loading indicators during file upload and query processing.
    - Improve display of search results (e.g., show snippets, highlight matches).
    - Add better visual feedback for errors.

## Low Priority / Future Ideas

- [ ] **Expand File Type Support:** Add reliable text extraction for `.docx`, `.xlsx`, etc. in `file_handlers.py`.
- [ ] **Implement Specialized Agents:**
    - Finance Agent
    - Learning Agent
    - Travel Agent
- [ ] **User Authentication/Accounts:** If needed for multi-user scenarios.
- [ ] **Testing:** Add unit and integration tests. 