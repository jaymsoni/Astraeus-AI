// API endpoints
const API_BASE_URL = 'http://localhost:5000';
const API_ENDPOINTS = {
    upload: `${API_BASE_URL}/upload`,
    query: `${API_BASE_URL}/query`
};

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const documentList = document.getElementById('documentList');
const chatMessages = document.getElementById('chatMessages');
const queryInput = document.getElementById('queryInput');
const sendQueryBtn = document.getElementById('sendQuery');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
sendQueryBtn.addEventListener('click', handleQuery);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleQuery();
});

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.style.borderColor = '#2c3e50';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.style.borderColor = '#ccc';
    
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
        const response = await fetch(API_ENDPOINTS.upload, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showStatus('success', `Successfully uploaded ${file.name}`);
            addDocumentToList(file.name);
        } else {
            showStatus('error', `Failed to upload ${file.name}: ${result.message}`);
        }
    } catch (error) {
        showStatus('error', `Error uploading ${file.name}: ${error.message}`);
    }
}

// Query Handlers
async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Add user message to chat
    addMessage(query, 'user');
    queryInput.value = '';
    
    try {
        const response = await fetch(API_ENDPOINTS.query, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // Add AI response to chat
            const aiResponse = formatResults(result.results);
            addMessage(aiResponse, 'ai');
        } else {
            addMessage(`Error: ${result.message}`, 'error');
        }
    } catch (error) {
        addMessage(`Error: ${error.message}`, 'error');
    }
}

// UI Helpers
function showStatus(type, message) {
    const statusDiv = document.createElement('div');
    statusDiv.className = `status ${type}`;
    statusDiv.textContent = message;
    uploadStatus.appendChild(statusDiv);
    
    // Remove status message after 5 seconds
    setTimeout(() => statusDiv.remove(), 5000);
}

function addDocumentToList(filename) {
    const docItem = document.createElement('div');
    docItem.className = 'document-item';
    docItem.innerHTML = `
        <span>${filename}</span>
        <span class="document-status">✓ Processed</span>
    `;
    documentList.appendChild(docItem);
}

function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = content;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatResults(results) {
    if (!results.length) return "I couldn't find any relevant information in the documents.";
    
    return results.map(result => result.content).join('\n\n');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
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
