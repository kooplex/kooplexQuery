import sqlite3
import os
from typing import List, Dict, Optional
import streamlit as st

DB_PATH = "database_servers.sqlite"

class DatabaseServerManager:
    """Manage SQLite database for storing database server information."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_database()
        self.database_server = {
            'url': None,
            'hostname': "localhost",
            'port': 5432,
            'database': "postgres",
            'user': "postgres",
            'password': "",
            'type': "postgresql",
            'schema': "public",
            'title': "Untitled"}
        self.chat_database_server = {        
            'chat_hostname': "localhost",
            'chat_port': 5432,
            'chat_database': "postgres",
            'chat_user': "postgres",
            'chat_password': "",
            'chat_schema': "public",
        }    
    
    def init_database(self):
        """Initialize SQLite database for storing database server information."""
        if os.path.exists(self.db_path):
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table to store database server information
        cursor.execute("""
            CREATE TABLE database_servers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                hostname TEXT,
                port INTEGER,
                database TEXT,
                user TEXT,
                password TEXT,
                type TEXT,
                schema TEXT,
                title TEXT,
                chat_hostname TEXT,
                chat_port INTEGER,
                chat_database TEXT,
                chat_user TEXT,
                chat_password TEXT,
                chat_schema TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def save_database_server(self) -> int:
        """Save database server information. Returns the ID of the inserted record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO database_servers 
            (url, hostname, port, database, user, password, type, schema, title, chat_hostname, chat_port, chat_database, chat_user, chat_password, chat_schema)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.database_server['url'], 
              self.database_server['hostname'], 
              self.database_server['port'], 
              self.database_server['database'], 
              self.database_server['user'], 
              self.database_server['password'], 
              self.database_server['type'], 
              self.database_server['schema'], 
              self.database_server['title']),
              self.chat_database_server['chat_hostname'], 
              self.chat_database_server['chat_port'], 
              self.chat_database_server['chat_database'], 
              self.chat_database_server['chat_user'], 
              self.chat_database_server['chat_password'], 
              self.chat_database_server['chat_schema'] )        
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        return record_id
    
    def get_database_server(self, server_id: int) -> Optional[Dict]:
        """Retrieve a specific database server by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM database_servers WHERE id = ?", (server_id,))
        db_row = cursor.fetchone()
        # put this info into self.database_server
        if db_row:
            self.database_server = {
                'url': db_row['url'],
                'hostname': db_row['hostname'],
                'port': db_row['port'],
                'database': db_row['database'],
                'user': db_row['user'],
                'password': db_row['password'],
                'type': db_row['type'],
                'schema': db_row['schema'],
                'title': db_row['title'],
                'chat_hostname': db_row['chat_hostname'],
                'chat_port': db_row['chat_port'],
                'chat_database': db_row['chat_database'],
                'chat_user': db_row['chat_user'],
                'chat_password': db_row['chat_password'],
                'chat_schema': db_row['chat_schema']
            }
        conn.close()
        
        return True
    
    def get_all_database_servers(self) -> List[Dict]:
        """Retrieve all database servers."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT url, hostname, port, database, user, password, type, schema, title, chat_hostname, chat_port, chat_database, chat_user, chat_password, chat_schema FROM database_servers ")
        db_rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(db_row) for db_row in db_rows]
    
    def delete_database_server(self, server_id: int) -> bool:
        """Delete a database server by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM database_servers WHERE id = ?", (server_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success


    # Pop up dialogs
    # @st.dialog("Database Configuration")
    def Database_configuration(self):
        st.html("<span class='big-dialog'></span>")
        def text_field(label, columns=None, **input_params):
            c1, c2 = st.columns(2)

            # Display field name with some alignment
            c1.markdown("####")
            c1.markdown(label)

            # Sets a default key parameter to avoid duplicate key errors
            input_params.setdefault("key", label)

            # Forward text input parameters
            return c2.text_input("", **input_params)
        # if st.session_state.get('custom_db_config') is None:
        #     st.session_state['custom_db_config'] = {
        #         "hostname": os.getenv("DB_HOST", "localhost"),
        #         "port": os.getenv("DB_PORT", "5432"),
        #         "database": os.getenv("DB_DATABASE", "mydatabase"),
        #         "db_user": os.getenv("DB_USER", "myuser"),
        #         "db_password": os.getenv("DB_PASSWORD", "mypassword"),
        #         "db_type": os.getenv("DB_TYPE", "postgres"),
        #         "db_schema": os.getenv("DB_SCHEMA", "distilled"),
        #         "db_title": os.getenv("DB_TITLE", "My Database")
        #     }
        # if st.session_state.get('custom_db_chat_config') is None:
        #     st.session_state['custom_db_config'] = {
        #         "hostname": os.getenv("CHAT_HOST", "localhost"),
        #         "port": os.getenv("CHAT_PORT", "5432"),
        #         "database": os.getenv("CHAT_DATABASE", "mydatabase"),
        #         "db_user": os.getenv("CHAT_USER", "myuser"),
        #         "db_password": os.getenv("CHAT_PASSWORD", "mypassword"),
        #         "db_schema": os.getenv("CHAT_SCHEMA", "distilled"),
        #     }
        # Open fileds to input database connection details, and update the connection when submitted
        st.file_uploader(label="Upload config file (json)", type="json", key="config_file")
        # Load configuration 
        database_configs = self.get_all_database_servers()
        selected_db = st.selectbox("Select Database config",
                database_configs,
                key="selected_config",
                format_func=lambda x:f"{x.title} - {x.type}",
                # on_change=on_model_selection_change,
                )
        if not selected_db:
            selected_db = self.get_all_database_servers()[0] if self.get_all_database_servers() else {'url': None,
             'hostname': "localhost",
             'port': 5432,
             'database': "postgres",
             'user': "postgres",
             'password': "",
             'type': "postgresql",
             'schema': "public",
             'title': "Untitled"}
        # Type manually connection details to connect to a database, and update the connection when submitted
        col1, col2 = st.columns(2)
        with st.form("db_info_form"):
            with col1:
                st.subheader("Database server")
                # st.text_input(label="Host", value=st.session_state['custom_db_config'].get('hostname'), key="db_host")
                text_field(label="Url", value=selected_db.get('url'), key="db_url")
                text_field(label="Host", value=selected_db.get('hostname'), key="db_host")
                text_field(label="User", value=selected_db.get('user'), key="db_user")
                text_field(label="Password", value="", key="db_password", type="password")
                text_field(label="Port", value=selected_db.get('port'), key="db_port")
                text_field(label="Database Name", value=selected_db.get('database'), key="db_name")
                text_field(label="Schema", value=selected_db.get('schema'), key="schema")
                text_field(label="Type", value=selected_db.get('type'), key="db_type")
                text_field(label="Database Title", value=selected_db.get('title'), key="db_title")
                    # if st.form_submit_button("Connect"):
                        # tmp_db = {
                            # "hostname": st.session_state.db_host,
                            # "port": st.session_state.db_port,
                            # "database": st.session_state.db_name,
                            # "db_user": st.session_state.db_user,
                            # "db_password": st.session_state.db_password,
                            # "db_type": st.session_state.db_type,
                            # "schema": st.session_state.schema,
                            # "db_title": st.session_state.db_title
                        # }
                        # tmp_config = st.session_state['custom_db_config']
                        # tmp_config.pop('db_title', None)
                        # st.session_state['motor'].db_source = st.session_state['motor']._dbtarget_init(tmp_config)
                        # st.success("Database connection updated!")
            with col2:
                st.subheader("Metadata server")
                text_field(label="Host", value=selected_db.get('hostname'), key="chat_host")
                text_field(label="User", value=selected_db.get('db_user'), key="chat_user")
                text_field(label="Password", value="", key="chat_password", type="password")
                text_field(label="Port", value=selected_db.get('port'), key="chat_port")
                text_field(label="Database", value=selected_db.get('database'), key="chat_database")
            if st.form_submit_button("Connect"):
                        self.chat_database_server = {
                            "hostname": st.session_state.db_host,
                            "url": st.session_state.db_url,
                            "port": st.session_state.db_port,
                            "database": st.session_state.db_name,
                            "user": st.session_state.db_user,
                            "password": st.session_state.db_password,
                            "type": st.session_state.db_type,
                            "schema": st.session_state.schema,
                            "title": st.session_state.db_title
                        }
                        self.chat_database_server = {
                            "chat_hostname": st.session_state.chat_host,
                            "chat_port": st.session_state.chat_port,
                            "chat_database": st.session_state.chat_database,
                            "chat_user": st.session_state.chat_user,
                            "chat_password": st.session_state.chat_password,
                            "chat_schema": st.session_state.chat_schema,
                        }
                        st.session_state['motor'].db_server = st.session_state['motor']._dbtarget_init(self.database_server)
                        st.session_state['motor'].db_chat = st.session_state['motor']._dbchat_init(self.chat_database_server)
                        st.success("Database connection updated!")
