import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ModelSelector from './ModelSelector';
import ChatInput from './ChatInput';
import '../styles/Chat.css';

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [conversationId, setConversationId] = useState(null);
    const [selectedModels, setSelectedModels] = useState({
        chat: 'gemini-2.0-flash',
        embedding: 'models/embedding-001'
    });

    const handleModelChange = async (type, modelId) => {
        console.log('Chat: Model change requested:', { type, modelId });
        try {
            const response = await axios.post('/api/models', {
                model_type: type,
                model_id: modelId
            });
            console.log('Chat: Model change response:', response.data);
            if (response.data.status === 'success') {
                setSelectedModels(prev => ({
                    ...prev,
                    [type]: modelId
                }));
            }
        } catch (error) {
            console.error('Chat: Error changing model:', error);
        }
    };

    const handleSendMessage = async (message) => {
        if (!message.trim()) return;

        setIsLoading(true);
        const userMessage = { role: 'user', content: message };
        setMessages(prev => [...prev, userMessage]);

        try {
            const response = await axios.post('/query', {
                query: message,
                conversation_id: conversationId,
                model_id: selectedModels.chat
            });

            if (response.data.status === 'success') {
                const assistantMessage = {
                    role: 'assistant',
                    content: response.data.response
                };
                setMessages(prev => [...prev, assistantMessage]);
                if (!conversationId) {
                    setConversationId(response.data.conversation_id);
                }
            }
        } catch (error) {
            console.error('Error sending message:', error);
            const errorMessage = {
                role: 'assistant',
                content: 'Sorry, there was an error processing your message. Please try again.'
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="chat-container">
            <div className="chat-header">
                <h1>Chat</h1>
                <ModelSelector onModelChange={handleModelChange} />
            </div>
            <div className="chat-messages">
                {messages.map((message, index) => (
                    <div key={index} className={`message ${message.role}`}>
                        {message.content}
                    </div>
                ))}
                {isLoading && (
                    <div className="message assistant typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                )}
            </div>
            <div className="chat-input-container">
                <ChatInput onSendMessage={handleSendMessage} />
            </div>
        </div>
    );
};

export default Chat; 