// API endpoints
const API_BASE_URL = 'http://localhost:5000';
const API_ENDPOINTS = {
    upload: `${API_BASE_URL}/upload`,
    query: `${API_BASE_URL}/query`,
    documents: `${API_BASE_URL}/documents` // Base path for document operations
};

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const documentList = document.getElementById('documentList');
const chatMessages = document.getElementById('chatMessages');
const queryInput = document.getElementById('queryInput');
const sendQueryBtn = document.getElementById('sendQuery');
const loadingOverlay = document.createElement('div'); // Create a loading overlay element
loadingOverlay.className = 'loading-overlay';
loadingOverlay.innerHTML = '<div class="spinner"></div><p>Processing...</p>';
document.body.appendChild(loadingOverlay);

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
sendQueryBtn.addEventListener('click', handleQuery);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleQuery();
});

// Loading state management
function showLoading(message = 'Processing...') {
    loadingOverlay.querySelector('p').textContent = message;
    loadingOverlay.classList.add('active');
}

function hideLoading() {
    loadingOverlay.classList.remove('active');
}

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

async function handleFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Add metadata about the file type
    const metadata = {
        type: file.type,
        name: file.name,
        size: file.size
    };
    formData.append('metadata', JSON.stringify(metadata));
    
    try {
        showLoading(`Uploading ${file.name}...`);
        showStatus('info', `Uploading ${file.name}...`);
        
        const response = await fetch(API_ENDPOINTS.upload, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        hideLoading();
        
        if (result.status === 'success') {
            showStatus('success', `Successfully uploaded ${file.name}`);
            addDocumentToList(file.name);
        } else {
            showStatus('error', `Failed to upload ${file.name}: ${result.message}`);
        }
    } catch (error) {
        hideLoading();
        showStatus('error', `Error uploading ${file.name}: ${error.message}`);
    }
}

// Query Handlers
async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query) return;
    
    addMessage(query, 'user');
    queryInput.value = '';
    queryInput.focus();
    
    try {
        showLoading('Thinking...'); // Updated loading message
        
        const response = await fetch(API_ENDPOINTS.query, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                query,
                threshold: 0.5 // Keep threshold for retrieval
            })
        });
        
        const result = await response.json();
        hideLoading();
        
        if (result.status === 'success') {
            // Display the generated answer from the LLM
            if (result.answer) {
                addMessage(result.answer, 'ai');
            } else {
                // Fallback if no answer provided for some reason
                 addMessage("I couldn't generate an answer based on the documents.", 'ai');
            }
            
            // Optional: Display source documents (could be a separate section or toggle)
            // console.log("Source documents:", result.results);
            // if (result.results && result.results.length > 0) {
            //     const sources = formatResults(result.results); // Use existing formatter
            //     addMessage(sources, 'sources'); // Add a new message type for sources?
            // }
            
        } else {
            // Display specific error message from backend if available
            addMessage(result.message || 'An error occurred while processing your query.', 'error');
        }
    } catch (error) { 
        hideLoading();
        // Display network or other frontend errors
        addMessage(`Network or processing error: ${error.message}`, 'error');
    }
}

// Document Deletion Handler
async function handleDelete(event) {
    const button = event.target;
    const docId = button.dataset.docId;
    const documentItem = button.closest('.document-item');

    if (!docId || !documentItem) {
        console.error("Could not find document ID or item for deletion.");
        showStatus('error', 'Could not initiate deletion. Please refresh.');
        return;
    }

    // Optional: Confirm before deleting
    if (!confirm(`Are you sure you want to delete ${docId}?`)) {
        return;
    }

    try {
        showLoading(`Deleting ${docId}...`);
        showStatus('info', `Deleting ${docId}...`);
        
        const response = await fetch(`${API_ENDPOINTS.documents}/${encodeURIComponent(docId)}`, {
            method: 'DELETE'
        });

        const result = await response.json();
        hideLoading();

        if (response.ok && result.status === 'success') {
            documentItem.remove();
            showStatus('success', result.message || `Successfully deleted ${docId}`);
        } else {
            console.error("Deletion failed:", result);
            showStatus('error', result.message || `Failed to delete ${docId}.`);
        }
    } catch (error) {
        hideLoading();
        console.error("Error during deletion request:", error);
        showStatus('error', `Error deleting ${docId}: ${error.message}`);
    }
}

// UI Helpers
function showStatus(type, message) {
    const statusDiv = document.createElement('div');
    statusDiv.className = `status ${type}`;
    statusDiv.textContent = message;
    uploadStatus.appendChild(statusDiv);
    
    // Auto-remove status message after 5 seconds
    setTimeout(() => {
        statusDiv.classList.add('fade-out');
        setTimeout(() => statusDiv.remove(), 500);
    }, 5000);
}

function addDocumentToList(filename) {
    console.log("Adding document to list:", filename);
    
    const docItem = document.createElement('div');
    docItem.className = 'document-item';
    
    // Create document name container
    const nameContainer = document.createElement('div');
    nameContainer.className = 'doc-name';
    nameContainer.textContent = filename;
    
    // Create delete button - make it more visible
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.className = 'delete-button';
    deleteBtn.dataset.docId = filename;
    deleteBtn.addEventListener('click', handleDelete);
    
    // Clear any existing content and append the new elements
    docItem.innerHTML = '';
    docItem.appendChild(nameContainer);
    docItem.appendChild(deleteBtn);
    
    documentList.appendChild(docItem);
    
    console.log("Document added with delete button:", filename);
}

function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    // Keep basic HTML rendering for newlines
    if (type === 'ai' || type === 'user') {
        // Basic sanitization (replace potential HTML tags, allow <br>)
        const sanitizedContent = String(content).replace(/</g, "&lt;").replace(/>/g, "&gt;");
        messageDiv.innerHTML = sanitizedContent.replace(/\n/g, '<br>');
    } else if (type === 'sources') {
        // If displaying sources directly, allow the HTML from formatResults
        messageDiv.innerHTML = content; 
    } else {
        // For error messages, just use textContent
        messageDiv.textContent = content;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatResults(results) {
    if (!results || !results.length) return "";
    
    let formattedResponse = '<div class="search-results">';
    formattedResponse += '<h4 class="sources-title">Sources Considered:</h4>'; // Add a title
    
    results.sort((a, b) => b.score - a.score);
    
    results.forEach((result, index) => {
        const confidencePercent = Math.round(result.score * 100);
        let snippet = result.content;
        if (snippet.length > 200) { // Shorter snippet for sources view
            snippet = snippet.substring(0, 200) + '...';
        }
        // Sanitize snippet before inserting as HTML
        const sanitizedSnippet = snippet.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, '<br>');
        
        formattedResponse += `
            <div class="result-item source-item">
                <div class="result-header">
                    <span class="result-title">Document: ${result.id}</span>
                    <span class="result-confidence">Relevance: ${confidencePercent}%</span>
                </div>
                <div class="result-content">${sanitizedSnippet}</div>
            </div>
        `;
    });
    
    formattedResponse += '</div>';
    return formattedResponse;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Add drop zone event listeners for better UX
    dropZone.addEventListener('dragenter', () => dropZone.classList.add('drag-over'));
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    
    // Check if backend is available
    fetch(API_BASE_URL)
        .then(response => response.json())
        .then(result => {
            if (result.status === 'success') {
                showStatus('success', 'Connected to backend successfully');
            }
        })
        .catch(error => {
            showStatus('error', 'Failed to connect to backend. Please ensure the server is running.');
        });
});
