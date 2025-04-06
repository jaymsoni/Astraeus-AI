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

// State
let currentConversationId = null;
let isProcessing = false;

// Add source visibility preference
let showSources = false;

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
sendQueryBtn.addEventListener('click', handleQuery);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleQuery();
    }
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
    if (isProcessing || !queryInput.value.trim()) return;

    const query = queryInput.value.trim();
    queryInput.value = '';
    
    // Store query for potential source fetching later
    window.lastResponseQuery = query;
    
    // Add user message to chat
    addMessage('user', query);
    
    // Show typing indicator
    showTypingIndicator();
    
    isProcessing = true;
    
    try {
        const response = await fetch(API_ENDPOINTS.query, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                query: query,
                conversation_id: currentConversationId,
                threshold: 0.6,      // Default threshold
                rerank: true,        // Enable reranking by default
                rerank_strategy: 'hybrid', // Default strategy
                use_hybrid: true,    // Enable hybrid search by default
                hybrid_ratio: 0.7,   // Default weight for semantic search
                show_sources: showSources // Only send sources if user wants to see them
            }),
        });

        const data = await response.json();
        
        if (data.status === 'success') {
            // Store the response and source count for later use
            window.lastResponse = data;
            window.lastResponseSourceCount = data.source_count || 0;
            
            // Display search results if available and sources are shown
            if (showSources && data.results && data.results.length > 0) {
                displaySearchResults(data.results, data.search_params);
            }
            
            currentConversationId = data.conversation_id;
            addMessage('assistant', data.answer);
        } else {
            addMessage('assistant', 'I apologize, but I encountered an error while processing your request.');
        }
    } catch (error) { 
        console.error('Error:', error);
        addMessage('assistant', 'I apologize, but something went wrong while processing your request.');
    } finally {
        hideTypingIndicator();
        isProcessing = false;
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

// Configure marked.js options
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true,
    headerIds: true,
    mangle: false,
    pedantic: false,
    smartLists: true,
    smartypants: true
});

// Custom renderer for better markdown handling
const renderer = new marked.Renderer();

// Enhance heading rendering
renderer.heading = function(text, level) {
    const escapedText = text.toLowerCase().replace(/[^\w]+/g, '-');
    return `
        <h${level} id="${escapedText}" class="markdown-heading">
            ${text}
        </h${level}>
    `;
};

// Enhance blockquote rendering
renderer.blockquote = function(quote) {
    return `<blockquote class="markdown-quote">${quote}</blockquote>`;
};

// Enhance list rendering
renderer.list = function(body, ordered) {
    const type = ordered ? 'ol' : 'ul';
    return `<${type} class="markdown-list">${body}</${type}>`;
};

// Update marked configuration with custom renderer
marked.use({ renderer });

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Handle markdown formatting for assistant messages
    if (role === 'assistant') {
        const md = window.markdownit({
            breaks: true,
            linkify: true,
            typographer: true,
            highlight: function (str, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(lang, str).value;
                    } catch (__) {}
                }
                return ''; // use external default escaping
            }
        });
        contentDiv.innerHTML = md.render(content);
    } else {
        // Simple text for user messages
        contentDiv.textContent = content;
    }
    
    messageDiv.appendChild(contentDiv);
    
    // Add source toggle button for assistant messages with sources
    if (role === 'assistant' && window.lastResponseSourceCount > 0) {
        const sourceToggle = document.createElement('button');
        sourceToggle.className = 'source-toggle';
        sourceToggle.textContent = showSources ? 'Hide Sources' : 'Show Sources';
        sourceToggle.onclick = function() {
            toggleSources();
            // Update all toggle buttons
            document.querySelectorAll('.source-toggle').forEach(btn => {
                btn.textContent = showSources ? 'Hide Sources' : 'Show Sources';
            });
        };
        messageDiv.appendChild(sourceToggle);
    }
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom of chat
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator message assistant';
    indicator.innerHTML = `
        <div class="message-content">
            <div class="dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const indicator = chatMessages.querySelector('.typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Toggle source visibility function
function toggleSources() {
    showSources = !showSources;
    
    // If sources are now visible and we have a lastResponse, fetch and show them
    if (showSources && window.lastResponse && window.lastResponseQuery) {
        fetchAndDisplaySources(window.lastResponseQuery, window.lastResponse.conversation_id);
    } else {
        // Hide sources if they're showing
        const resultsContainer = document.getElementById('searchResults');
        if (resultsContainer) {
            resultsContainer.style.display = 'none';
        }
    }
}

// Function to fetch sources for a query
async function fetchAndDisplaySources(query, conversationId) {
    try {
        const response = await fetch(API_ENDPOINTS.query, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                conversation_id: conversationId,
                show_sources: true,
                threshold: 0.6,
                rerank: true,
                rerank_strategy: 'hybrid',
                use_hybrid: true,
                hybrid_ratio: 0.7
            }),
        });

        const data = await response.json();
        
        if (data.status === 'success' && data.results && data.results.length > 0) {
            displaySearchResults(data.results, data.search_params);
        }
    } catch (error) {
        console.error('Error fetching sources:', error);
    }
}

// Function to display search results
function displaySearchResults(results, searchParams) {
    if (!results || results.length === 0) return;
    
    // Create a container for search results if it doesn't exist
    let resultsContainer = document.getElementById('searchResults');
    if (!resultsContainer) {
        resultsContainer = document.createElement('div');
        resultsContainer.id = 'searchResults';
        resultsContainer.className = 'search-results';
        chatMessages.appendChild(resultsContainer);
    } else {
        // Clear existing results
        resultsContainer.innerHTML = '';
    }
    
    // Make sure it's visible
    resultsContainer.style.display = 'block';
    
    // Add header with search parameters
    const headerDiv = document.createElement('div');
    headerDiv.className = 'search-results-header';
    headerDiv.innerHTML = `
        <h3>Source Documents (${results.length})</h3>
        <div class="search-params">
            <span>Threshold: ${searchParams?.threshold || 0.6}</span>
            <span>Reranking: ${searchParams?.rerank ? (searchParams.rerank_strategy || 'hybrid') : 'disabled'}</span>
            <span>Hybrid Search: ${searchParams?.use_hybrid ? `Enabled (${Math.round(searchParams.hybrid_ratio * 100)}% semantic)` : 'Disabled'}</span>
        </div>
    `;
    resultsContainer.appendChild(headerDiv);
    
    // Group results by source document
    const groupedResults = {};
    results.forEach(result => {
        const sourceId = result.parent_id || result.id;
        const sourceTitle = result.parent_title || sourceId;
        
        if (!groupedResults[sourceId]) {
            groupedResults[sourceId] = {
                title: sourceTitle,
                items: []
            };
        }
        
        groupedResults[sourceId].items.push(result);
    });
    
    // Create result elements
    Object.values(groupedResults).forEach(group => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'result-source';
        
        const titleDiv = document.createElement('div');
        titleDiv.className = 'source-title';
        titleDiv.textContent = group.title;
        sourceDiv.appendChild(titleDiv);
        
        group.items.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            // Add a CSS class based on retrieval method
            const retrievalMethod = result.retrieval_method || 'unknown';
            resultItem.classList.add(`retrieval-${retrievalMethod}`);
            
            const scoreSpan = document.createElement('span');
            scoreSpan.className = 'result-score';
            const score = Math.round(result.score * 100);
            
            // Add retrieval method to the score display
            scoreSpan.innerHTML = `
                <span class="score-value">${score}% match</span>
                <span class="retrieval-method">${retrievalMethod}</span>
            `;
            
            const snippetDiv = document.createElement('div');
            snippetDiv.className = 'result-snippet';
            snippetDiv.textContent = result.snippet || result.content.substring(0, 200) + '...';
            
            resultItem.appendChild(scoreSpan);
            resultItem.appendChild(snippetDiv);
            sourceDiv.appendChild(resultItem);
        });
        
        resultsContainer.appendChild(sourceDiv);
    });
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

    // Update the welcome message with markdown
    const welcomeMessage = `# Welcome to Astraeus! 👋

I'm your AI assistant, ready to help you understand and work with your documents. Here's what I can do:

* 📝 **Answer questions** about your uploaded documents
* 📊 **Analyze and summarize** document content
* 🔍 **Find specific information** across multiple files
* 💡 **Provide insights** and connections between documents

You can start by:
1. Uploading some documents using the upload area above
2. Asking me questions about your documents
3. Requesting specific analysis or summaries

Need help getting started? Just ask!`;

    // Remove the default welcome message
    const existingWelcome = chatMessages.querySelector('.message.assistant');
    if (existingWelcome) {
        existingWelcome.remove();
    }

    // Add the new markdown welcome message
    addMessage('assistant', welcomeMessage);
});
