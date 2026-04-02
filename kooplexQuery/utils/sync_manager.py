import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading
import queue

logger = logging.getLogger(__name__)


class VectorStoreSyncManager:
    """
    Manages automatic synchronization between PostgreSQL metadata database
    and Chroma vector store.
    
    Whenever data changes in the metadata database, it's automatically 
    synced to the corresponding vectorstore collection.
    """
    
    def __init__(self, db_chat=None, vectorstore=None):
        """
        Initialize the sync manager.
        
        Args:
            db_chat: DBChat instance for database access
            vectorstore: VectorStore instance for vector store access
        """
        self.db_chat = db_chat
        self.vectorstore = vectorstore
        self.sync_log = []
        self.sync_enabled = True
        self._lock = threading.Lock()
        
    def set_db_chat(self, db_chat):
        """Set the DBChat instance after initialization."""
        self.db_chat = db_chat
        
    def set_vectorstore(self, vectorstore):
        """Set the VectorStore instance after initialization."""
        self.vectorstore = vectorstore
    
    def enable_sync(self):
        """Enable synchronization."""
        self.sync_enabled = True
        logger.info("Vector store synchronization enabled")
        
    def disable_sync(self):
        """Disable synchronization temporarily."""
        self.sync_enabled = False
        logger.info("Vector store synchronization disabled")
    
    def _validate_instances(self):
        """Validate that both db_chat and vectorstore are set."""
        if not self.db_chat or not self.vectorstore:
            raise RuntimeError(
                "Both db_chat and vectorstore must be initialized before syncing. "
                "Use set_db_chat() and set_vectorstore() to set them."
            )
    
    def sync_knowledge(self, reference: str, content: str) -> bool:
        """
        Sync knowledge entry to vectorstore after it's saved to database.
        Called automatically whenever save_knowledge() is called in DBChat.
        
        Args:
            reference: The reference key for the knowledge
            content: The knowledge content
            
        Returns:
            bool: True if sync succeeded, False otherwise
        """
        if not self.sync_enabled:
            return False
            
        try:
            self._validate_instances()
            
            # Determine which collection based on reference type
            collection_map = {
                'schema': 'schema',
                'instruction': 'docs',
                'data_descriptor': 'docs',
                'data_reference': 'docs',
                'advice': 'advices',
                'example': 'examples'
            }
            
            collection_name = collection_map.get(reference, 'docs')
            
            # Create metadata with reference info
            metadatas = [{
                'reference': reference,
                'type': collection_name,
                'synced_at': datetime.now().isoformat()
            }]
            
            # For schema, we split on CREATE statements
            if reference == 'schema':
                self.vectorstore.load_split_add_text(
                    content,
                    collection_name=collection_name,
                    split_on="CREATE"
                )
            else:
                # For other types, add as single text
                if not self.vectorstore._check_similarity(content, 
                                                         self.vectorstore._select_collection_by_name(collection_name)):
                    collection = self.vectorstore._select_collection_by_name(collection_name)
                    collection.add_texts(texts=[content], metadatas=metadatas)
            
            self._log_sync('knowledge', reference, True)
            logger.info(f"Synced knowledge '{reference}' to vectorstore")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing knowledge '{reference}': {e}")
            self._log_sync('knowledge', reference, False, str(e))
            return False
    
    def sync_example(self, question_id: int, question_content: str, sql: str, 
                    is_public: bool = True) -> bool:
        """
        Sync example (question-SQL pair) to vectorstore.
        Called automatically when a new validated example is added.
        
        Args:
            question_id: ID of the question
            question_content: The question text
            sql: The SQL query
            is_public: Whether the example is public
            
        Returns:
            bool: True if sync succeeded, False otherwise
        """
        if not self.sync_enabled or not is_public:
            return False
            
        try:
            self._validate_instances()
            
            item = {
                'question': question_content,
                'sql': sql
            }
            self.vectorstore.add_to_examples(item)
            
            self._log_sync('example', f'question_{question_id}', True)
            logger.info(f"Synced example {question_id} to vectorstore")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing example {question_id}: {e}")
            self._log_sync('example', f'question_{question_id}', False, str(e))
            return False
    
    def sync_table_description(self, table_name: str, description: str) -> bool:
        """
        Sync table description to vectorstore.
        Called automatically when table descriptions are updated.
        
        Args:
            table_name: Name of the table
            description: Description text
            
        Returns:
            bool: True if sync succeeded, False otherwise
        """
        if not self.sync_enabled:
            return False
            
        try:
            self._validate_instances()
            
            texts = [description]
            metadatas = [{
                'Table': table_name,
                'type': 'table_description',
                'synced_at': datetime.now().isoformat()
            }]
            
            self.vectorstore.add_to_docs(texts=texts, metadatas=metadatas)
            
            self._log_sync('table_description', table_name, True)
            logger.info(f"Synced table description for '{table_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing table description for '{table_name}': {e}")
            self._log_sync('table_description', table_name, False, str(e))
            return False
    
    def sync_column_description(self, column_name: str, table_name: str, 
                               description: str, data_type: str = '') -> bool:
        """
        Sync column description to vectorstore.
        Called automatically when column descriptions are updated.
        
        Args:
            column_name: Name of the column
            table_name: Name of the table
            description: Description text
            data_type: Data type of the column
            
        Returns:
            bool: True if sync succeeded, False otherwise
        """
        if not self.sync_enabled:
            return False
            
        try:
            self._validate_instances()
            
            # Format as "Column Name (DataType) - Description"
            full_description = f"{column_name} ({data_type})" if data_type else column_name
            full_description += f" - {description}" if description else ""
            
            texts = [full_description]
            metadatas = [{
                'Column': column_name,
                'Table': table_name,
                'type': 'column_description',
                'synced_at': datetime.now().isoformat()
            }]
            
            self.vectorstore.add_to_docs(texts=texts, metadatas=metadatas)
            
            self._log_sync('column_description', f'{table_name}.{column_name}', True)
            logger.info(f"Synced column description for '{table_name}.{column_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing column description for '{table_name}.{column_name}': {e}")
            self._log_sync('column_description', f'{table_name}.{column_name}', False, str(e))
            return False
    
    def batch_sync_examples(self, limit: int = None) -> int:
        """
        Batch sync all public examples from database to vectorstore.
        Useful for initial sync or recovery.
        
        Args:
            limit: Limit number of examples to sync
            
        Returns:
            int: Number of examples synced
        """
        if not self.sync_enabled:
            return 0
            
        try:
            self._validate_instances()
            
            examples = self.db_chat.fetch_all_examples()
            keys, data = examples
            
            synced_count = 0
            for row in data:
                # Convert row to dict using keys
                row_dict = dict(zip(keys, row))
                
                # Only sync public examples
                if row_dict.get('public'):
                    item = {
                        'question': row_dict.get('question_content'),
                        'sql': row_dict.get('sql')
                    }
                    self.vectorstore.add_to_examples(item)
                    synced_count += 1
            
            self._log_sync('batch_examples', 'global', True, f"Synced {synced_count} examples")
            logger.info(f"Batch synced {synced_count} public examples to vectorstore")
            return synced_count
            
        except Exception as e:
            logger.error(f"Error in batch sync of examples: {e}")
            self._log_sync('batch_examples', 'global', False, str(e))
            return 0
    
    def batch_sync_knowledge(self) -> int:
        """
        Batch sync all knowledge entries from database to vectorstore.
        
        Returns:
            int: Number of knowledge entries synced
        """
        if not self.sync_enabled:
            return 0
            
        try:
            self._validate_instances()
            
            # Fetch all knowledge entries from database
            # Note: This requires a method in DBChat to fetch all knowledge
            # For now, we'll load common references
            common_references = ['schema', 'instruction', 'data_descriptor', 'data_reference']
            
            synced_count = 0
            for reference in common_references:
                content = self.db_chat.load_knowledge(reference)
                if content:
                    self.sync_knowledge(reference, content)
                    synced_count += 1
            
            self._log_sync('batch_knowledge', 'global', True, f"Synced {synced_count} entries")
            logger.info(f"Batch synced {synced_count} knowledge entries to vectorstore")
            return synced_count
            
        except Exception as e:
            logger.error(f"Error in batch sync of knowledge: {e}")
            self._log_sync('batch_knowledge', 'global', False, str(e))
            return 0
    
    def resync_all(self) -> Dict[str, int]:
        """
        Complete resync of all data from database to vectorstore.
        Useful for recovery or migration.
        
        Returns:
            dict: Summary of synced items by type
        """
        logger.info("Starting full resync of vectorstore...")
        
        results = {
            'knowledge': self.batch_sync_knowledge(),
            'examples': self.batch_sync_examples(),
        }
        
        logger.info(f"Full resync completed: {results}")
        self._log_sync('resync', 'full', True, f"Synced {sum(results.values())} items")
        
        return results
    
    def _log_sync(self, sync_type: str, target: str, success: bool, 
                 details: str = '') -> None:
        """
        Log a sync operation for audit trail.
        
        Args:
            sync_type: Type of sync operation
            target: Target item/reference
            success: Whether sync succeeded
            details: Additional details
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': sync_type,
            'target': target,
            'success': success,
            'details': details
        }
        
        with self._lock:
            self.sync_log.append(log_entry)
            # Keep only last 1000 entries
            if len(self.sync_log) > 1000:
                self.sync_log = self.sync_log[-1000:]
    
    def get_sync_log(self, limit: int = 50) -> List[Dict]:
        """
        Get recent sync log entries.
        
        Args:
            limit: Number of recent entries to return
            
        Returns:
            list: Recent sync log entries
        """
        with self._lock:
            return self.sync_log[-limit:]
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """
        Get synchronization statistics.
        
        Returns:
            dict: Statistics about sync operations
        """
        with self._lock:
            if not self.sync_log:
                return {
                    'total_operations': 0,
                    'successful': 0,
                    'failed': 0,
                    'success_rate': 0.0
                }
            
            total = len(self.sync_log)
            successful = sum(1 for entry in self.sync_log if entry['success'])
            failed = total - successful
            
            return {
                'total_operations': total,
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / total) * 100 if total > 0 else 0.0,
                'sync_enabled': self.sync_enabled
            }


class DBChatWithSync:
    """
    Wrapper around DBChat that automatically syncs changes to vectorstore.
    
    Usage:
        db_chat = DBChatWithSync(hostname, port, database, schema, db_user, db_password)
        db_chat.set_vectorstore(vectorstore_instance)
        db_chat.save_knowledge(...)  # Will auto-sync
    """
    
    def __init__(self, hostname, port, database, schema, db_user, db_password, 
                 generated_callback=lambda c: None):
        """Initialize with database connection parameters."""
        from kooplexQuery.db_chat import DBChat
        
        self._db_chat = DBChat(
            hostname=hostname,
            port=port,
            database=database,
            schema=schema,
            db_user=db_user,
            db_password=db_password,
            generated_callback=generated_callback
        )
        
        self.sync_manager = VectorStoreSyncManager(db_chat=self._db_chat)
    
    def set_vectorstore(self, vectorstore):
        """Set the vectorstore for automatic syncing."""
        self.sync_manager.set_vectorstore(vectorstore)
    
    def save_knowledge(self, reference: str, content: str):
        """Save knowledge to database and auto-sync to vectorstore."""
        result = self._db_chat.save_knowledge(reference, content)
        
        # Sync to vectorstore
        self.sync_manager.sync_knowledge(reference, content)
        
        return result
    
    def save_query(self, session_id, question_content, sql, question_type='user', 
                  public=True):
        """Save query to database and auto-sync if it's a validated example."""
        result = self._db_chat.save_query(
            session_id=session_id,
            question_content=question_content,
            sql=sql,
            question_type=question_type,
            public=public
        )
        
        # Get the question ID that was just created
        # This would require modifying save_query to return the ID
        # For now, sync is handled separately
        
        return result
    
    def validate_question(self, question_id):
        """Validate a question and auto-sync to vectorstore."""
        result = self._db_chat.validate_question(question_id)
        
        # Fetch the validated example and sync it
        try:
            q = self._db_chat.engine.execute(
                f"""SELECT q.content, a.sql FROM question q 
                   JOIN query a ON q.id = a.question_id 
                   WHERE q.id = {question_id}"""
            )
            question_content, sql = q.fetchone()
            self.sync_manager.sync_example(question_id, question_content, sql, True)
        except Exception as e:
            logger.error(f"Error syncing validated question {question_id}: {e}")
        
        return result
    
    # Delegate all other methods to the underlying DBChat instance
    def __getattr__(self, name):
        return getattr(self._db_chat, name)
