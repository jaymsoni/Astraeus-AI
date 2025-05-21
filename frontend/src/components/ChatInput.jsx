import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const ChatInput = ({ onSendMessage }) => {
    const [message, setMessage] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [mentionStart, setMentionStart] = useState(-1);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const inputRef = useRef(null);
    const suggestionsRef = useRef(null);

    // Fetch document suggestions when '@' is typed
    const fetchSuggestions = async (query) => {
        try {
            const response = await axios.get('/documents/suggestions');
            if (response.data.status === 'success') {
                // Filter and sort suggestions based on query
                const filtered = response.data.suggestions
                    .filter(doc => 
                        doc.name.toLowerCase().includes(query.toLowerCase()) ||
                        (doc.summary && doc.summary.toLowerCase().includes(query.toLowerCase()))
                    )
                    .sort((a, b) => {
                        // Sort by relevance to query
                        const aStartsWith = a.name.toLowerCase().startsWith(query.toLowerCase());
                        const bStartsWith = b.name.toLowerCase().startsWith(query.toLowerCase());
                        if (aStartsWith && !bStartsWith) return -1;
                        if (!aStartsWith && bStartsWith) return 1;
                        return a.name.localeCompare(b.name);
                    });
                setSuggestions(filtered);
                setShowSuggestions(filtered.length > 0);
                setSelectedIndex(0); // Reset selection when new suggestions appear
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            setShowSuggestions(false);
        }
    };

    // Handle input changes
    const handleInputChange = (e) => {
        const value = e.target.value;
        setMessage(value);

        // Check for '@' mention
        const lastAtIndex = value.lastIndexOf('@');
        if (lastAtIndex !== -1) {
            const query = value.substring(lastAtIndex + 1);
            setMentionStart(lastAtIndex);
            fetchSuggestions(query);
        } else {
            setShowSuggestions(false);
            setMentionStart(-1);
        }
    };

    // Handle keyboard navigation
    const handleKeyDown = (e) => {
        if (!showSuggestions) return;

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex(prev => 
                    prev < suggestions.length - 1 ? prev + 1 : prev
                );
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex(prev => prev > 0 ? prev - 1 : prev);
                break;
            case 'Enter':
                e.preventDefault();
                if (suggestions[selectedIndex]) {
                    handleSuggestionClick(suggestions[selectedIndex]);
                }
                break;
            case 'Escape':
                setShowSuggestions(false);
                break;
        }
    };

    // Handle suggestion selection
    const handleSuggestionClick = (doc) => {
        if (mentionStart !== -1) {
            const beforeMention = message.substring(0, mentionStart);
            const afterMention = message.substring(message.indexOf(' ', mentionStart) || message.length);
            const newMessage = `${beforeMention}@${doc.name}${afterMention}`;
            setMessage(newMessage);
            setShowSuggestions(false);
            // Focus input after selection
            inputRef.current.focus();
        }
    };

    // Handle message submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (message.trim()) {
            onSendMessage(message);
            setMessage('');
            setShowSuggestions(false);
        }
    };

    // Close suggestions when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (inputRef.current && !inputRef.current.contains(event.target)) {
                setShowSuggestions(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);

    // Scroll selected suggestion into view
    useEffect(() => {
        if (suggestionsRef.current && selectedIndex >= 0) {
            const selectedElement = suggestionsRef.current.children[selectedIndex];
            if (selectedElement) {
                selectedElement.scrollIntoView({
                    block: 'nearest',
                    behavior: 'smooth'
                });
            }
        }
    }, [selectedIndex]);

    return (
        <div className="chat-input-container" ref={inputRef}>
            <form onSubmit={handleSubmit} className="chat-input-form">
                <div className="mention-container">
                    <input
                        ref={inputRef}
                        type="text"
                        value={message}
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        placeholder="Type your message... Use @ to mention documents"
                        className="chat-input"
                    />
                    {showSuggestions && suggestions.length > 0 && (
                        <div className="suggestions-dropdown" ref={suggestionsRef}>
                            {suggestions.map((doc, index) => (
                                <div
                                    key={doc.id}
                                    className={`suggestion-item ${index === selectedIndex ? 'selected' : ''}`}
                                    onClick={() => handleSuggestionClick(doc)}
                                    onMouseEnter={() => setSelectedIndex(index)}
                                >
                                    <div className="suggestion-content">
                                        <span className="doc-name">
                                            <span className="mention-symbol">@</span>
                                            {doc.name}
                                        </span>
                                        {doc.summary && (
                                            <span className="doc-summary">{doc.summary}</span>
                                        )}
                                    </div>
                                    {index === selectedIndex && (
                                        <div className="selection-indicator">↵</div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
                <button type="submit" className="send-button">
                    Send
                </button>
            </form>
        </div>
    );
};

export default ChatInput; 