// API endpoints
const API_BASE_URL = 'http://localhost:5000';
const API_ENDPOINTS = {
    upload: `${API_BASE_URL}/upload`,
    query: `${API_BASE_URL}/query`,
    documents: `${API_BASE_URL}/documents` // Base path for document operations
};

// File type configuration
const allowed_file_types = [
    'text/plain',
    'text/markdown',
    'application/pdf',
    'application/json',
    'text/csv',
    'application/xml',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // docx
    'application/msword', // doc
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // xlsx
    'application/vnd.ms-excel', // xls
    'text/html'
];

const allowed_extensions = [
    'txt', 'md', 'pdf', 'json', 'csv', 'xml', 'yaml', 'yml',
    'doc', 'docx', 'xls', 'xlsx', 'html', 'htm', 'rtf'
];

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const documentList = document.getElementById('documentList');
const chatMessages = document.getElementById('chatMessages');
const queryInput = document.getElementById('queryInput');
const sendQueryBtn = document.getElementById('sendQuery');

// Document Panel Elements
const documentPanel = document.getElementById('documentPanel');
const documentPanelToggle = document.getElementById('documentPanelToggle');
const closePanelBtn = document.getElementById('closePanelBtn');
const panelDocumentList = document.getElementById('panelDocumentList');
const panelFileInput = document.getElementById('panelFileInput');

// Create loading overlay
const loadingOverlay = document.createElement('div');
loadingOverlay.className = 'loading-overlay';
loadingOverlay.innerHTML = `
    <div class="loading-spinner"></div>
    <div class="loading-message">Loading...</div>
`;
document.body.appendChild(loadingOverlay);

// App state variables
let showSources = false;
let currentConversationId = null;
window.lastResponse = null;
window.lastResponseQuery = null;
window.lastResponseSourceCount = 0;

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

// Document Panel Event Listeners
documentPanelToggle.addEventListener('click', toggleDocumentPanel);
closePanelBtn.addEventListener('click', closeDocumentPanel);
panelFileInput.addEventListener('change', handlePanelFileSelect);

// Loading state management
function showLoading(message = 'Loading...') {
    loadingOverlay.className = 'loading-overlay visible';
    loadingOverlay.innerHTML = `
        <div class="loading-spinner"></div>
        <div class="loading-message">${message}</div>
    `;
}

function hideLoading() {
    loadingOverlay.className = 'loading-overlay';
}

// Drag and drop handlers
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
    
    if (e.dataTransfer.files.length) {
        handleFiles(e.dataTransfer.files);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length) {
        handleFiles(e.target.files);
    }
}

function handleFiles(files) {
    // Process each file
    Array.from(files).forEach(file => {
        if (allowed_file_types.includes(file.type) || 
            allowed_extensions.some(ext => file.name.toLowerCase().endsWith(`.${ext}`))) {
            uploadFile(file);
        } else {
            showStatus('error', `File type not allowed: ${file.name}`);
        }
    });
}

async function uploadFile(file) {
    try {
        showLoading(`Uploading ${file.name}...`);
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('metadata', JSON.stringify({ 
            source: 'web_upload',
            description: 'User uploaded document'
        }));
        
        const response = await fetch(API_ENDPOINTS.upload, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showStatus('success', `${file.name} uploaded successfully`);
            // Refresh document list instead of just adding to UI
            fetchDocuments();
            // Also refresh the panel documents if the panel is open
            if (documentPanel.classList.contains('open')) {
                loadDocumentsToPanel();
            }
        } else {
            throw new Error(result.message || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showStatus('error', `Error uploading ${file.name}: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Handle query submission
async function handleQuery() {
    const query = queryInput.value.trim();
    
    if (!query) {
        return; // Don't process empty queries
    }
    
    try {
        // Show user's message
    addMessage('user', query);
    
        // Clear input
        queryInput.value = '';
        
        // Show typing indicator while we wait for a response
    showTypingIndicator();
    
        // Prepare request data
        const requestData = {
            query: query,
            conversation_id: currentConversationId
        };
        
        const response = await fetch(API_ENDPOINTS.query, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        if (data.status === 'success') {
            // Store conversation ID for future messages
            currentConversationId = data.conversation_id;
            
            // Add assistant's response
            addMessage('assistant', data.answer);
            
            // Store response for potential source viewing
            window.lastResponse = data;
            window.lastResponseQuery = query;
            window.lastResponseSourceCount = data.source_count || 0;
            
            // If showSources is true, immediately fetch and display sources
            if (showSources && data.source_count > 0) {
                fetchAndDisplaySources(query, data.conversation_id);
            }
        } else {
            throw new Error(data.message || 'Failed to process query');
        }
    } catch (error) { 
        console.error('Error processing query:', error);
        hideTypingIndicator();
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
    }
}

// Document Deletion Handler
async function handleDelete(event) {
    const button = event.target;
    const docId = button.dataset.docId;

    if (!docId) {
        console.error("Could not find document ID for deletion.");
        return;
    }

    // Confirm delete
    if (!confirm(`Are you sure you want to delete "${docId}"?`)) {
        return;
    }

    try {
        showLoading(`Deleting ${docId}...`);
        
        const response = await fetch(`${API_ENDPOINTS.documents}/${encodeURIComponent(docId)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const result = await response.json();
        
        if (result.status === 'success') {
            showStatus('success', `${docId} deleted successfully`);
            // Refresh document list instead of removing just one element
            fetchDocuments();
        } else {
            throw new Error(result.message || 'Deletion failed');
        }
    } catch (error) {
        console.error("Error during deletion request:", error);
        showStatus('error', `Error deleting ${docId}: ${error.message}`);
    } finally {
        hideLoading();
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

// Fetch all documents when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded, initializing app...');
    
    // Initialize with a welcome message
    addMessage('assistant', `# Welcome to Astraeus AI Assistant 🚀

I'm your AI assistant, ready to help you understand and work with your documents. Here's what I can do:

* 📝 **Answer questions** about your uploaded documents
* 📊 **Analyze and summarize** document content
* 🔍 **Find information** across multiple documents
* 💡 **Provide insights** and connections between documents

Get started by:
1. Uploading some documents using the upload area above
2. Asking me questions about your documents
`);

    // Add "Manage Documents" button to the documents section
    const documentsSection = document.querySelector('.documents-section');
    const documentsSectionHeader = documentsSection.querySelector('h2');
    
    const manageDocumentsBtn = document.createElement('button');
    manageDocumentsBtn.className = 'manage-documents-btn';
    manageDocumentsBtn.innerHTML = '📑 Manage All Documents';
    manageDocumentsBtn.addEventListener('click', toggleDocumentPanel);
    
    documentsSectionHeader.appendChild(manageDocumentsBtn);

    // Attach event listeners
    try {
        attachEventListeners();
        console.log('Event listeners attached successfully');
    } catch (error) {
        console.error('Error attaching event listeners:', error);
    }
    
    // Fetch and display all documents
    try {
        fetchDocuments();
        console.log('Fetching documents...');
    } catch (error) {
        console.error('Error fetching documents:', error);
        showStatus('error', `Failed to load documents: ${error.message}`);
    }
});

// Attach all event listeners
function attachEventListeners() {
    // Add drop zone event listeners for better UX
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('dragenter', () => dropZone.classList.add('drag-over'));
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    
    // File input change event
    fileInput.addEventListener('change', handleFileSelect);
    
    // Query input events
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleQuery();
        }
    });
    
    // Send query button click
    sendQueryBtn.addEventListener('click', handleQuery);
    
    // Document panel events
    documentPanelToggle.addEventListener('click', toggleDocumentPanel);
    closePanelBtn.addEventListener('click', closeDocumentPanel);
    panelFileInput.addEventListener('change', handlePanelFileSelect);
    
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
}

// Fetch all documents from the server
async function fetchDocuments() {
    try {
        showLoading('Loading documents...');
        
        const response = await fetch(API_ENDPOINTS.documents, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Documents response:', data); // Debug log
        
        if (data.status === 'success') {
            // Clear existing document list
            documentList.innerHTML = '';
            
            // Add each document to the list
            if (data.documents && Array.isArray(data.documents) && data.documents.length > 0) {
                data.documents.forEach(doc => {
                    if (doc) { // Make sure doc is defined
                        addDocumentWithSummary(doc);
                    } else {
                        console.error('Received undefined document in list');
                    }
                });
                
                // Update the document panel if it's open
                if (documentPanel.classList.contains('open')) {
                    loadDocumentsToPanel();
                }
            } else {
                documentList.innerHTML = '<div class="no-documents">No documents uploaded yet</div>';
            }
        } else {
            showStatus('error', data.message || 'Failed to fetch documents');
        }
    } catch (error) {
        console.error('Error fetching documents:', error);
        showStatus('error', `Error fetching documents: ${error.message}`);
        // Show an empty state when there's an error
        documentList.innerHTML = '<div class="no-documents">Could not load documents. Please try again.</div>';
    } finally {
        hideLoading();
    }
}

// Add a document with its summary to the document list
function addDocumentWithSummary(doc) {
    const docItem = document.createElement('div');
    docItem.className = 'document-item';
    
    // Create document container with filename
    const docContainer = document.createElement('div');
    docContainer.className = 'doc-container';
    
    // Create document header with name
    const nameContainer = document.createElement('div');
    nameContainer.className = 'doc-name';
    nameContainer.textContent = doc.filename;
    
    // Create summary container
    const summaryContainer = document.createElement('div');
    summaryContainer.className = 'doc-summary';
    
    // Create summary toggle button
    const summaryToggle = document.createElement('button');
    summaryToggle.className = 'summary-toggle';
    summaryToggle.textContent = 'Show Summary';
    summaryToggle.addEventListener('click', () => {
        // Toggle summary visibility
        if (summaryContainer.style.display === 'none' || !summaryContainer.style.display) {
            summaryContainer.style.display = 'block';
            summaryToggle.textContent = 'Hide Summary';
        } else {
            summaryContainer.style.display = 'none';
            summaryToggle.textContent = 'Show Summary';
        }
    });
    
    // Set summary content
    summaryContainer.textContent = doc.summary || 'No summary available';
    summaryContainer.style.display = 'none'; // Hidden by default
    
    // Create delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.className = 'delete-button';
    deleteBtn.dataset.docId = doc.id;
    deleteBtn.addEventListener('click', handleDelete);
    
    // Add file info
    const fileInfo = document.createElement('div');
    fileInfo.className = 'file-info';
    fileInfo.textContent = `${formatFileSize(doc.size)} • ${doc.type || 'unknown type'}`;
    
    // Build the document item structure
    docContainer.appendChild(nameContainer);
    docContainer.appendChild(fileInfo);
    docContainer.appendChild(summaryToggle);
    docContainer.appendChild(summaryContainer);
    
    docItem.appendChild(docContainer);
    docItem.appendChild(deleteBtn);
    
    documentList.appendChild(docItem);
}

// Helper to format file size
function formatFileSize(bytes) {
    if (!bytes || isNaN(bytes)) return 'Unknown size';
    
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// Document Panel Functions
function toggleDocumentPanel() {
    documentPanel.classList.toggle('open');
    if (documentPanel.classList.contains('open')) {
        // Refresh document list when opening the panel
        loadDocumentsToPanel();
    }
}

function closeDocumentPanel() {
    documentPanel.classList.remove('open');
}

function handlePanelFileSelect(e) {
    if (e.target.files.length) {
        handleFiles(e.target.files);
        // Clear the input value so the same file can be selected again
        e.target.value = '';
    }
}

function loadDocumentsToPanel() {
    // Clear existing content
    panelDocumentList.innerHTML = '<div class="panel-loading">Loading documents...</div>';
    
    // Fetch documents from server
    fetch(API_ENDPOINTS.documents, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => {
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            displayDocumentsInPanel(data.documents);
        } else {
            throw new Error(data.message || 'Failed to fetch documents');
        }
    })
    .catch(error => {
        console.error('Error loading documents to panel:', error);
        panelDocumentList.innerHTML = `
            <div class="panel-empty-state">
                Could not load documents. ${error.message}
            </div>
        `;
    });
}

function displayDocumentsInPanel(documents) {
    // Clear previous content
    panelDocumentList.innerHTML = '';
    
    if (!documents || documents.length === 0) {
        panelDocumentList.innerHTML = `
            <div class="panel-empty-state">
                No documents uploaded yet. Use the upload area below to add documents.
            </div>
        `;
        return;
    }
    
    // Sort documents by upload date (newest first)
    documents.sort((a, b) => {
        return new Date(b.uploaded_at || 0) - new Date(a.uploaded_at || 0);
    });
    
    // Create document items
    documents.forEach(doc => {
        const docItem = document.createElement('div');
        docItem.className = 'panel-document-item';
        
        // Create document title
        const titleDiv = document.createElement('div');
        titleDiv.className = 'panel-document-title';
        titleDiv.textContent = doc.filename;
        
        // Create document info
        const infoDiv = document.createElement('div');
        infoDiv.className = 'panel-document-info';
        
        const formattedSize = formatFileSize(doc.size);
        const formattedDate = formatDate(doc.uploaded_at);
        infoDiv.textContent = `${formattedSize} • ${doc.type || 'Unknown'} • Uploaded: ${formattedDate}`;
        
        // Add summary toggle and content if available
        let summaryDiv = null;
        if (doc.summary && doc.summary !== 'No summary available') {
            // Create toggle
            const summaryToggle = document.createElement('button');
            summaryToggle.className = 'panel-document-action';
            summaryToggle.textContent = 'Show Summary';
            
            // Create summary content (hidden by default)
            summaryDiv = document.createElement('div');
            summaryDiv.className = 'panel-document-summary';
            summaryDiv.textContent = doc.summary;
            summaryDiv.style.display = 'none';
            
            // Toggle functionality
            summaryToggle.addEventListener('click', () => {
                if (summaryDiv.style.display === 'none') {
                    summaryDiv.style.display = 'block';
                    summaryToggle.textContent = 'Hide Summary';
                } else {
                    summaryDiv.style.display = 'none';
                    summaryToggle.textContent = 'Show Summary';
                }
            });
            
            // Add to document item
            docItem.appendChild(titleDiv);
            docItem.appendChild(infoDiv);
            docItem.appendChild(summaryToggle);
            docItem.appendChild(summaryDiv);
        } else {
            // No summary available
            docItem.appendChild(titleDiv);
            docItem.appendChild(infoDiv);
        }
        
        // Create document actions
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'panel-document-actions';
        
        // Add actions buttons
        const useChatBtn = document.createElement('button');
        useChatBtn.className = 'panel-document-action';
        useChatBtn.textContent = 'Ask Question';
        useChatBtn.addEventListener('click', () => {
            // Focus on query input and suggest asking about this document
            queryInput.value = `Tell me about ${doc.filename}`;
            queryInput.focus();
            closeDocumentPanel();
        });
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'panel-document-action delete';
        deleteBtn.textContent = 'Delete';
        deleteBtn.dataset.docId = doc.id;
        deleteBtn.addEventListener('click', handlePanelDelete);
        
        // Add action buttons to actions div
        actionsDiv.appendChild(useChatBtn);
        actionsDiv.appendChild(deleteBtn);
        
        // Add actions to document item
        docItem.appendChild(actionsDiv);
        
        // Add the document item to the panel list
        panelDocumentList.appendChild(docItem);
    });
}

// Helper function to format date
function formatDate(dateString) {
    if (!dateString) return 'Unknown date';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
        });
    } catch (e) {
        return 'Unknown date';
    }
}

// Handle document deletion from panel
async function handlePanelDelete(event) {
    const docId = event.target.dataset.docId;
    
    if (!docId) {
        console.error("Could not find document ID for deletion.");
        return;
    }
    
    // Confirm delete
    if (!confirm(`Are you sure you want to delete "${docId}"?`)) {
        return;
    }
    
    try {
        showLoading(`Deleting ${docId}...`);
        
        const response = await fetch(`${API_ENDPOINTS.documents}/${encodeURIComponent(docId)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showStatus('success', `${docId} deleted successfully`);
            // Refresh document lists in both panel and main area
            fetchDocuments();
            loadDocumentsToPanel();
        } else {
            throw new Error(result.message || 'Deletion failed');
        }
    } catch (error) {
        console.error("Error during deletion request:", error);
        showStatus('error', `Error deleting ${docId}: ${error.message}`);
    } finally {
        hideLoading();
    }
}
