"""
Conversation Memory Management System
Handles storage, retrieval, and formatting of conversation history for chatbot continuity.
"""

import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from collections import defaultdict, deque
import threading


class ConversationManager:
    """
    Manages conversation history with multiple storage backends:
    - In-memory (fastest, lost on restart)
    - SQLite database (persistent, good for production)
    - JSON file (persistent, simple for development)
    """
    
    def __init__(self, 
                 storage_type: str = "memory",  # "memory", "sqlite", "json"
                 max_history: int = 10,
                 db_path: str = "conversations.db",
                 json_path: str = "conversations.json",
                 cleanup_after_hours: int = 24):
        """
        Initialize conversation manager.
        
        Args:
            storage_type: "memory", "sqlite", or "json"
            max_history: Maximum number of exchanges to keep per conversation
            db_path: Path to SQLite database file
            json_path: Path to JSON storage file
            cleanup_after_hours: Auto-cleanup conversations older than this
        """
        self.storage_type = storage_type
        self.max_history = max_history
        self.cleanup_after_hours = cleanup_after_hours
        self._lock = threading.Lock()  # Thread safety for concurrent requests
        
        if storage_type == "memory":
            self._init_memory_storage()
        elif storage_type == "sqlite":
            self._init_sqlite_storage(db_path)
        elif storage_type == "json":
            self._init_json_storage(json_path)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def _init_memory_storage(self):
        """Initialize in-memory storage using deques for efficient FIFO operations."""
        self.conversations = defaultdict(lambda: deque(maxlen=self.max_history))
        self.conversation_metadata = {}
    
    def _init_sqlite_storage(self, db_path: str):
        """Initialize SQLite database for persistent storage."""
        self.db_path = db_path
        self.conversations = {}  # Cache for frequently accessed conversations
        
        # Create database and table if they don't exist
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    response_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    INDEX(conversation_id, timestamp)
                )
            """)
            conn.commit()
    
    def _init_json_storage(self, json_path: str):
        """Initialize JSON file storage."""
        self.json_path = json_path
        self.conversations = {}
        
        # Load existing conversations if file exists
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations = {
                        conv_id: deque(exchanges, maxlen=self.max_history)
                        for conv_id, exchanges in data.items()
                    }
            except (json.JSONDecodeError, FileNotFoundError):
                self.conversations = {}
    
    def add_exchange(self, 
                    conversation_id: str, 
                    question: str, 
                    answer: str,
                    response_id: str = None,
                    metadata: Dict = None) -> None:
        """
        Add a question-answer exchange to the conversation history.
        
        Args:
            conversation_id: Unique identifier for the conversation
            question: User's question
            answer: Bot's response
            response_id: Unique identifier for this specific response
            metadata: Additional information (retrieval flags, confidence scores, etc.)
        """
        with self._lock:
            exchange = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'response_id': response_id,
                'metadata': metadata or {}
            }
            
            if self.storage_type == "memory":
                self._add_to_memory(conversation_id, exchange)
            elif self.storage_type == "sqlite":
                self._add_to_sqlite(conversation_id, exchange)
            elif self.storage_type == "json":
                self._add_to_json(conversation_id, exchange)
    
    def _add_to_memory(self, conversation_id: str, exchange: Dict):
        """Add exchange to memory storage."""
        self.conversations[conversation_id].append(exchange)
        self.conversation_metadata[conversation_id] = {
            'last_activity': exchange['timestamp'],
            'total_exchanges': len(self.conversations[conversation_id])
        }
    
    def _add_to_sqlite(self, conversation_id: str, exchange: Dict):
        """Add exchange to SQLite storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversations 
                (conversation_id, question, answer, response_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation_id,
                exchange['question'],
                exchange['answer'],
                exchange['response_id'],
                json.dumps(exchange['metadata'])
            ))
            conn.commit()
        
        # Update cache
        if conversation_id in self.conversations:
            self.conversations[conversation_id].append(exchange)
    
    def _add_to_json(self, conversation_id: str, exchange: Dict):
        """Add exchange to JSON storage."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = deque(maxlen=self.max_history)
        
        self.conversations[conversation_id].append(exchange)
        self._save_json()
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """
        Retrieve conversation history for a specific conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            List of exchange dictionaries, ordered chronologically
        """
        with self._lock:
            if self.storage_type == "memory":
                return list(self.conversations.get(conversation_id, []))
            elif self.storage_type == "sqlite":
                return self._get_from_sqlite(conversation_id)
            elif self.storage_type == "json":
                return list(self.conversations.get(conversation_id, []))
    
    def _get_from_sqlite(self, conversation_id: str) -> List[Dict]:
        """Retrieve conversation from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT question, answer, response_id, timestamp, metadata
                FROM conversations 
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (conversation_id, self.max_history))
            
            exchanges = []
            for row in cursor:
                exchange = {
                    'timestamp': row['timestamp'],
                    'question': row['question'],
                    'answer': row['answer'],
                    'response_id': row['response_id'],
                    'metadata': json.loads(row['metadata'] or '{}')
                }
                exchanges.append(exchange)
            
            return list(reversed(exchanges))  # Return in chronological order
    
    def format_history_for_llm(self, 
                              history: List[Dict], 
                              max_tokens: int = 1000,
                              include_metadata: bool = False) -> str:
        """
        Format conversation history for LLM consumption.
        
        Args:
            history: List of exchange dictionaries
            max_tokens: Approximate maximum tokens to include (rough estimate)
            include_metadata: Whether to include retrieval metadata
            
        Returns:
            Formatted string ready for LLM input
        """
        if not history:
            return ""
        
        formatted_exchanges = []
        estimated_tokens = 0
        
        # Process from most recent to oldest
        for exchange in reversed(history):
            # Format the exchange
            formatted_exchange = f"Human: {exchange['question']}\nAssistant: {exchange['answer']}"
            
            if include_metadata and exchange.get('metadata'):
                meta = exchange['metadata']
                meta_info = []
                if meta.get('qna_retrieved'):
                    meta_info.append("QnA retrieved")
                if meta.get('pdf_retrieved'):
                    meta_info.append("PDF retrieved")
                if meta_info:
                    formatted_exchange += f" [Retrieved: {', '.join(meta_info)}]"
            
            # Rough token estimation (1 token ≈ 4 characters)
            exchange_tokens = len(formatted_exchange) // 4
            
            if estimated_tokens + exchange_tokens > max_tokens:
                break
                
            formatted_exchanges.insert(0, formatted_exchange)  # Insert at beginning
            estimated_tokens += exchange_tokens
        
        return "\n\n".join(formatted_exchanges)
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear all history for a specific conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            True if conversation existed and was cleared, False otherwise
        """
        with self._lock:
            if self.storage_type == "memory":
                if conversation_id in self.conversations:
                    del self.conversations[conversation_id]
                    if conversation_id in self.conversation_metadata:
                        del self.conversation_metadata[conversation_id]
                    return True
                return False
                
            elif self.storage_type == "sqlite":
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM conversations WHERE conversation_id = ?",
                        (conversation_id,)
                    )
                    deleted = cursor.rowcount > 0
                    conn.commit()
                    
                # Clear from cache
                if conversation_id in self.conversations:
                    del self.conversations[conversation_id]
                    
                return deleted
                
            elif self.storage_type == "json":
                if conversation_id in self.conversations:
                    del self.conversations[conversation_id]
                    self._save_json()
                    return True
                return False
    
    def list_active_conversations(self) -> List[str]:
        """
        List all active conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        if self.storage_type == "memory":
            return list(self.conversations.keys())
        elif self.storage_type == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT conversation_id FROM conversations")
                return [row[0] for row in cursor]
        elif self.storage_type == "json":
            return list(self.conversations.keys())
    
    def cleanup_old_conversations(self) -> int:
        """
        Remove conversations older than the specified time limit.
        
        Returns:
            Number of conversations cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=self.cleanup_after_hours)
        cutoff_iso = cutoff_time.isoformat()
        
        cleaned_count = 0
        
        if self.storage_type == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE timestamp < ?",
                    (cutoff_iso,)
                )
                cleaned_count = cursor.rowcount
                conn.commit()
        
        elif self.storage_type == "memory":
            conversations_to_remove = []
            for conv_id, metadata in self.conversation_metadata.items():
                if metadata['last_activity'] < cutoff_iso:
                    conversations_to_remove.append(conv_id)
            
            for conv_id in conversations_to_remove:
                del self.conversations[conv_id]
                del self.conversation_metadata[conv_id]
                cleaned_count += 1
        
        elif self.storage_type == "json":
            conversations_to_remove = []
            for conv_id, exchanges in self.conversations.items():
                if exchanges and exchanges[-1]['timestamp'] < cutoff_iso:
                    conversations_to_remove.append(conv_id)
            
            for conv_id in conversations_to_remove:
                del self.conversations[conv_id]
                cleaned_count += 1
            
            if cleaned_count > 0:
                self._save_json()
        
        return cleaned_count
    
    def _save_json(self):
        """Save conversations to JSON file."""
        # Convert deques to lists for JSON serialization
        data = {
            conv_id: list(exchanges)
            for conv_id, exchanges in self.conversations.items()
        }
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored conversations.
        
        Returns:
            Dictionary with conversation statistics
        """
        stats = {
            'total_conversations': 0,
            'total_exchanges': 0,
            'storage_type': self.storage_type,
            'max_history_per_conversation': self.max_history
        }
        
        if self.storage_type == "memory":
            stats['total_conversations'] = len(self.conversations)
            stats['total_exchanges'] = sum(len(exchanges) for exchanges in self.conversations.values())
            
        elif self.storage_type == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(DISTINCT conversation_id) FROM conversations")
                stats['total_conversations'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                stats['total_exchanges'] = cursor.fetchone()[0]
                
        elif self.storage_type == "json":
            stats['total_conversations'] = len(self.conversations)
            stats['total_exchanges'] = sum(len(exchanges) for exchanges in self.conversations.values())
        
        return stats