import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../styles/ModelSelector.css';

const ModelSelector = ({ onModelChange }) => {
    const [models, setModels] = useState({
        chat: 'gemini-2.0-flash',
        embedding: 'models/embedding-001'
    });
    const [availableModels, setAvailableModels] = useState({
        chat: [],
        embedding: []
    });

    useEffect(() => {
        // Fetch available models from backend
        const fetchModels = async () => {
            try {
                console.log('Fetching available models...');
                const response = await axios.get('/api/models');
                console.log('Received models:', response.data);
                if (response.data.status === 'success') {
                    setAvailableModels(response.data.models);
                }
            } catch (error) {
                console.error('Error fetching available models:', error);
            }
        };

        fetchModels();
    }, []);

    const handleModelChange = (type, modelId) => {
        console.log('Model change:', { type, modelId });
        setModels(prev => ({
            ...prev,
            [type]: modelId
        }));
        onModelChange(type, modelId);
    };

    const groupModelsByProvider = (models) => {
        const groups = {
            'Google': [],
            'Ollama': []
        };

        models.forEach(model => {
            if (model.id.startsWith('ollama/')) {
                groups['Ollama'].push(model);
            } else {
                groups['Google'].push(model);
            }
        });

        console.log('Grouped models:', groups);
        return groups;
    };

    const renderModelOptions = (models) => {
        const groupedModels = groupModelsByProvider(models);
        
        return Object.entries(groupedModels).map(([provider, providerModels]) => (
            <optgroup key={provider} label={provider}>
                {providerModels.map(model => (
                    <option key={model.id} value={model.id}>
                        {model.name}
                    </option>
                ))}
            </optgroup>
        ));
    };

    return (
        <div className="model-selector">
            <div className="model-selector-group">
                <label htmlFor="chat-model">Chat Model:</label>
                <select
                    id="chat-model"
                    value={models.chat}
                    onChange={(e) => handleModelChange('chat', e.target.value)}
                    className="model-select"
                >
                    {renderModelOptions(availableModels.chat)}
                </select>
            </div>
            <div className="model-selector-group">
                <label htmlFor="embedding-model">Embedding Model:</label>
                <select
                    id="embedding-model"
                    value={models.embedding}
                    onChange={(e) => handleModelChange('embedding', e.target.value)}
                    className="model-select"
                >
                    {renderModelOptions(availableModels.embedding)}
                </select>
            </div>
        </div>
    );
};

export default ModelSelector; 