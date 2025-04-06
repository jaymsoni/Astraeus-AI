from typing import List, Dict
import time

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add a message to the conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message = {
            'role': role,
            'content': content,
            'timestamp': time.time()
        }
        self.conversations[conversation_id].append(message)
    
    def get_conversation_history(self, conversation_id: str, max_messages: int = 5) -> List[Dict]:
        """Get the recent conversation history."""
        if conversation_id not in self.conversations:
            return []
        
        return self.conversations[conversation_id][-max_messages:]
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id] 