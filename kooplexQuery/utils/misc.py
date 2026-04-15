import sqlite3
import os
import json
from typing import List, Dict, Optional
import streamlit as st
import logging
from kooplexQuery.utils.vectorstore import VectorStore
import pandas as pd

logging.basicConfig(
    filename='/tmp//app.log', 
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)
logger.debug("START")

try:
    import yaml
except Exception:
    yaml = None

DB_PATH = "database_servers.sqlite"

def upload_schema_to_chat_db(schema_file):
    try:
        raw = schema_file.read()
        try:
            content = raw.decode("utf-8")
        except Exception:
            content = raw.decode("utf-8", errors="replace")
        st.session_state['motor'].db_chat.save_knowledge(reference='schema', content=content)
        st.session_state['meta_schema'] = content
        
        st.success("Database schema uploaded to chat database successfully!")
    except Exception as e:
        logger.error(f"Error uploading database schema: {e}")
        st.error("Failed to upload database schema.")



def _get_session_state_db_config() -> Dict:
    return dict({
        "database_server": {"url": st.session_state.database_server.get("url"),
        "hostname": st.session_state.database_server.get("hostname"),
        "user": st.session_state.database_server.get("user"),
        "port": st.session_state.database_server.get("port", 5432),
        "database": st.session_state.database_server.get("database"),
        "schema": st.session_state.database_server.get("schema", "public"),
        "type": st.session_state.database_server.get("type"),
        "title": st.session_state.database_server.get("title"),
        },
            "chat_database_server": {
        "hostname": st.session_state.chat_database_server.get("hostname"),
        "user": st.session_state.chat_database_server.get("user"),
        "port": st.session_state.chat_database_server.get("port", 5432),
        "database": st.session_state.chat_database_server.get("database"),
        "schema": st.session_state.chat_database_server.get("schema"),
        "password": st.session_state.chat_database_server.get("password"),  
            }
    })

class DatabaseServerManager:
    """Manage SQLite database for storing database server information."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_database()
        if 'database_server' not in st.session_state:
            st.write("Initializing database server configuration in session state.")
            st.session_state['database_server'] = {
                'url': None,
                'hostname': "localhost",
                'port': 5432,
                'database': "postgres",
                'user': "postgres",
                'password': "",
                'type': "postgresql",
                'schema': "public",
                'title': "Untitled"}
        if 'chat_database_server' not in st.session_state:
            st.session_state['chat_database_server'] = {        
                'hostname': "localhost",
                'port': 5432,
                'database': "postgres",
                'user': "postgres",
                'password': "",
                'schema': "public",
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
                db_url TEXT,
                db_hostname TEXT,
                db_port INTEGER,
                db_database TEXT,
                db_user TEXT,
                db_password TEXT,
                db_type TEXT,
                db_schema TEXT,
                db_title TEXT,
                chat_hostname TEXT,
                chat_port INTEGER,
                chat_database TEXT,
                chat_user TEXT,
                chat_password TEXT,
                chat_schema TEXT,
                UNIQUE(db_url, db_title)
                UNIQUE(db_hostname, db_database, db_schema)
            )
        """)
        conn.commit()
        conn.close()
    
    def save_database_server(self) -> int:
        """Save database server information. Returns the ID of the inserted or existing record."""
        db_cfg = st.session_state['database_server']
        chat_cfg = st.session_state['chat_database_server']

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO database_servers
            (db_url, db_hostname, db_port, db_database, db_user, db_password, db_type, db_schema, db_title, chat_hostname, chat_port, chat_database, chat_user, chat_password, chat_schema)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                db_cfg.get('url'),
                db_cfg.get('hostname'),
                db_cfg.get('port'),
                db_cfg.get('database'),
                db_cfg.get('user'),
                db_cfg.get('password'),
                db_cfg.get('type'),
                db_cfg.get('schema'),
                db_cfg.get('title'),
                chat_cfg.get('hostname'),
                chat_cfg.get('port'),
                chat_cfg.get('database'),
                chat_cfg.get('user'),
                chat_cfg.get('password'),
                chat_cfg.get('schema'),
            ),
        )
        conn.commit()

        if cursor.rowcount > 0:
            record_id = cursor.lastrowid
        else:
            if db_cfg.get('url'):
                cursor.execute("SELECT id FROM database_servers WHERE db_url = ?", (db_cfg.get('url'),))
            else:
                cursor.execute(
                    "SELECT id FROM database_servers WHERE db_hostname = ? AND db_database = ? AND db_schema = ?",
                    (db_cfg.get('hostname'), db_cfg.get('database'), db_cfg.get('schema')),
                )
            row = cursor.fetchone()
            record_id = row[0] if row else None

        conn.close()
        
        if record_id is None:
            raise RuntimeError("Failed to save database server configuration.")

        return int(record_id)
    
    def get_database_server(self, server_id: int) -> Optional[Dict]:
        """Retrieve a specific database server by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM database_servers WHERE id = ?", (server_id,))
        db_row = cursor.fetchone()
        # put this info into st.session_state['database_server']
        self._update_config(db_row, db_row) if db_row else None
      
        conn.close()
        
        return True
    
    def get_all_database_servers(self) -> List[Dict]:
        """Retrieve all database servers."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, db_url, db_hostname, db_port, db_database, db_user, db_password, db_type, db_schema, db_title, chat_hostname, chat_port, chat_database, chat_user, chat_password, chat_schema FROM database_servers ")
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

    def _update_config(self, db_cfg, chat_cfg):

        
        if db_cfg:
            try:
                for key in st.session_state['database_server'].keys():
                    if key in db_cfg:
                        st.session_state['database_server'][key] = str(db_cfg[key])
                    else:
                        st.session_state['database_server'][key] = str(db_cfg[f"db_{key}"])
            except Exception as e:
                logger.warning(f"Failed to update database server config with db_cfg: {e}. Attempting fallback key format.")
        if chat_cfg:
            try:
                for key in st.session_state['chat_database_server'].keys():
                    if key in chat_cfg:
                        st.session_state['chat_database_server'][key] = str(chat_cfg[key])
                    else:
                        st.session_state['chat_database_server'][key] = str(chat_cfg[f"chat_{key}"])
            except Exception as e:
                logger.warning(f"Failed to update chat database server config with chat_cfg: {e}. Attempting fallback key format.")

        logger.debug(f"Updated session state database_server config: {st.session_state['database_server']}")
        logger.debug(f"Updated session state chat_database_server config: {st.session_state['chat_database_server']}")

    def _connect(self):
        try:
            # Keep existing saved passwords when password fields are left empty.
            self._update_config(st.session_state, st.session_state)

            new_database_signature = (
                st.session_state.get("db_url") or "",
                st.session_state.get("db_hostname") or "",
                int(st.session_state.get("db_port") or 5432),
                st.session_state.get("db_database") or "",
                st.session_state.get("db_schema") or "",
                st.session_state.get("db_type") or "",
            )
            previous_database_signature = st.session_state.get("active_database_signature")

            if st.session_state.get('motor') is None:
                from kooplexQuery.motor import Motor
                st.session_state['motor'] = object.__new__(Motor)
                st.session_state['motor']._table_name_filter = '%'

            st.session_state['motor'].db_source = st.session_state['motor']._dbtarget_init(st.session_state['database_server'])
            st.session_state['motor'].db_chat = st.session_state['motor']._dbchat_init(st.session_state['chat_database_server'])
            st.session_state['motor']._ensure_sync_manager()
            st.session_state['motor_init_error'] = None

            

            # if not same_as_selected:
            try:
                self.save_database_server()
            except Exception as e:
                logger.warning(f"Failed to save database configuration: {e}")
                st.warning("Database connection was successful, but failed to save the configuration. Please check the logs for details.")

            if previous_database_signature != new_database_signature:
                vecstore = st.session_state.get("vecstore")
                if vecstore is not None:
                    try:
                        vecstore.reset()
                    except Exception as e:
                        logger.warning(f"Failed to reset existing vector store: {e}")

                st.session_state["vecstore"] = None
                st.session_state["statistics"] = None
                st.session_state["data_descriptor"] = None
                st.session_state["instruction"] = None
                # st.session_state["db_schema"] = None
                st.session_state["database_reference"] = None
                st.session_state["table_descriptions"] = None
                st.session_state["column_descriptions"] = None
                st.session_state["examples"] = None
                st.session_state['interface'] = "chat"
                st.session_state["active_database_signature"] = new_database_signature
                logger.info("Database changed; vector store reset for the newly loaded database")

            # Start a fresh chat session immediately after successful connect.
            st.session_state['db_config_set'] = True
            st.session_state['session_init_failed'] = False
            # Force prompt flow to create a fresh chat session for the (possibly reconfigured)
            # Motor instance so attributes like session_id are always present.
            st.session_state.pop('session', None)
            # Always trigger one-shot automatic sync into vectorstore after connect.
            st.session_state['pending_vectorstore_sync'] = True
                # st.success("Database connection updated. Existing configuration unchanged, so it was not saved again. New session started.")
                # st.success("Database connection updated, configuration saved, and a new session started.")
            st.rerun()
        except Exception as e:
            st.error(       
                f"Connection or session initialization failed: {e}. "
                "Please open Database settings and try connecting again."
            )
            raise

    # Pop up dialogs
    # @st.dialog("Database Configuration")
    def Database_configuration(self):
        st.html("<span class='big-dialog'></span>")
        st.markdown(
            """
            <style>
            div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
                gap: 0.5rem;
            }
            div[data-testid="stForm"] [data-testid="stMarkdownContainer"] p {
                margin-bottom: 0.2rem;
            }
            div[data-testid="stForm"] div[data-testid="stTextInput"] {
                margin-bottom: 0.15rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        def text_field(label, columns=None, **input_params):
            c1, c2 = st.columns(2)

            # Keep label and input aligned in a single visual row.
            c1.markdown(
                f"<div style='line-height:2.2rem; white-space:nowrap;'>{label}</div>",
                unsafe_allow_html=True,
            )

            # Sets a default key parameter to avoid duplicate key errors
            input_params.setdefault("key", label)

            widget_key = input_params.get("key")

            # Streamlit text_input requires string values in widget state.
            if widget_key is not None and widget_key in st.session_state:
                existing_value = st.session_state.get(widget_key)
                if existing_value is not None and not isinstance(existing_value, str):
                    st.session_state[widget_key] = str(existing_value)

            # Avoid setting both a default value and session state for the same key.
            if widget_key is not None and widget_key in st.session_state and "value" in input_params:
                input_params.pop("value")

            # Streamlit text_input expects a string value.
            if "value" in input_params:
                input_params["value"] = "" if input_params["value"] is None else str(input_params["value"])

            # Forward text input parameters
            input_params.setdefault("label_visibility", "hidden")
            return c2.text_input(label, **input_params)

       
        # Open fileds to input database connection details, and update the connection when submitted
        upload_col, select_col, download_delete_col = st.columns(3)
        with upload_col:
            uploaded_config_file = st.file_uploader(
                label="Upload config file (json/yaml)",
                type=["json", "yaml", "yml"],
                key="config_file",
            )

        if uploaded_config_file is not None:
            try:
                raw = uploaded_config_file.read()
                try:
                    config_text = raw.decode("utf-8")
                except Exception:
                    config_text = raw.decode("utf-8", errors="replace")

                file_name = (uploaded_config_file.name or "").lower()
                if file_name.endswith(".json"):
                    uploaded_config = json.loads(config_text)
                else:
                    if yaml is None:
                        raise ValueError("PyYAML is not available to parse YAML files.")
                    uploaded_config = yaml.safe_load(config_text)

                if not isinstance(uploaded_config, dict):
                    raise ValueError("Config file must contain a JSON/YAML object at the top level.")

                db_cfg = uploaded_config.get("database_server", uploaded_config)
                chat_cfg = uploaded_config.get("chat_database_server", uploaded_config)
                # print("Parsed db config:", db_cfg
                        # , "Parsed chat db config:", chat_cfg)
                self._update_config(db_cfg, chat_cfg)
                # print("Updated session state with uploaded config:", st.session_state['database_server'], st.session_state['chat_database_server'])
                self.save_database_server()
               
                st.success("Config file loaded successfully.")
                uploaded_config_file = None  # Clear the file uploader after successful upload
                st.session_state['selected_db_index'] = len(self.get_all_database_servers())-1 if self.get_all_database_servers() else 0
                # st.rerun()  # Refresh the page to update the selected config and form fields
            except Exception as e:
                # st.error(f"Failed to parse uploaded config: {e}")
                st.error(f"Config already exists or failed to save: {e}. Please check the file format and contents, and ensure it doesn't duplicate an existing configuration.")

        def on_model_selection_change():
             selected_config = st.session_state.get("selected_config")
             if selected_config is not None:
                self._update_config(selected_config, selected_config)
                #  sync_form_fields_from_config(selected_config)
                st.session_state.update({ key: selected_config.get(key, "") for key in selected_config.keys() })
                st.session_state['db_config_set'] = True
                st.session_state['pending_vectorstore_sync'] = True
                st.success("Database configuration loaded. Please click Connect to apply the configuration and start a new session.")
             else:
                st.warning("Selected database configuration not found.")
                st.session_state['db_config_set'] = False
                st.session_state['pending_vectorstore_sync'] = False

        # Load configuration 
        database_configs = self.get_all_database_servers()
        logger.debug(f"Available database configs: {len(database_configs)}")
        with select_col:
            selected_db = st.selectbox("Select Database config",
                    database_configs,
                    key="selected_config",
                    format_func=lambda x: f"{x.get('db_title', 'Untitled')} - {x.get('type', '')}",
                    index=st.session_state.get('selected_db_index', 0) if database_configs else 0,
                    on_change=on_model_selection_change,
                    placeholder='None available, please add a config'
                    )
            # if st.session_state.database_server.get("title") is None and database_configs:
                # self._update_config(st.session_state.selected_db, st.session_state.selected_db)
            self._update_config(st.session_state.selected_config, st.session_state.selected_config)
            st.session_state['selected_db_index'] = self.get_all_database_servers().index(st.session_state.selected_config) if st.session_state.selected_config in self.get_all_database_servers() else 0
            logger.info(f"Selected DB config: {st.session_state['selected_db_index']}")
            # st.stop()


        if yaml is not None:
            yaml_content = yaml.safe_dump(_get_session_state_db_config(), sort_keys=False, allow_unicode=False)
        else:
            # JSON is valid YAML 1.2, so this remains importable by YAML parsers.
            yaml_content = json.dumps(_get_session_state_db_config(), indent=2)

        with download_delete_col:
            download_title = str(_get_session_state_db_config()["database_server"].get("title") or "database_config")
            safe_download_title = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in download_title).strip("_")
            safe_download_title = safe_download_title or "database_config"
            st.download_button(
                label="Download config (yaml)",
                data=yaml_content,
                file_name=f"{safe_download_title}.yaml",
                mime="application/x-yaml",
                key="download_config_yaml",
            )

            if st.button("Delete config", key="delete_selected_config", disabled=self.get_all_database_servers() == [] ):
                st.session_state['selected_db_index'] = 0
                if self.delete_database_server(int(selected_db["id"])):
                    st.success("Selected database config deleted.")
                    st.rerun()
                else:
                    st.warning("Could not delete selected config.")

        # Type manually connection details to connect to a database, and update the connection when submitted
        
        with st.form("db_info_form"):
            if st.form_submit_button("Connect"):
                    self._connect()
                    st.success("Database connection updated, configuration saved, and a new session started.")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Database server")
                # Use session state value if it exists, otherwise use selected_db value (avoids Streamlit widget warning)
                text_field(label="Url", value=st.session_state.database_server.get("url"), key="db_url")
                text_field(label="Host", value= st.session_state.database_server.get("hostname"), key="db_hostname")
                text_field(label="User", value=st.session_state.database_server.get("user"), key="db_user")
                text_field(label="Password", value=st.session_state.database_server.get("password"), key="db_password", type="password")
                text_field(label="Port", value=st.session_state.database_server.get("port"), key="db_port")
                text_field(label="Database Name", value=st.session_state.database_server.get("database"), key="db_database")
                text_field(label="Schema", value=st.session_state.database_server.get("schema"), key="db_schema")
                text_field(label="Type", value= st.session_state.database_server.get("type"), key="db_type")
                text_field(label="Database Title", value= st.session_state.database_server.get("title"), key="db_title")
            with col2:
                st.subheader("Metadata server")
                text_field(label="Host", value=st.session_state.chat_database_server.get("hostname"), key="chat_hostname")
                text_field(label="User", value=st.session_state.chat_database_server.get("user"), key="chat_user")
                text_field(label="Password", value=st.session_state.chat_database_server.get("password"), key="chat_password", type="password")
                text_field(label="Port", value=st.session_state.chat_database_server.get("port"), key="chat_port")
                text_field(label="Database", value=st.session_state.chat_database_server.get("database"), key="chat_database")
                text_field(label="Schema", value=st.session_state.chat_database_server.get("schema"), key="chat_schema")
            
                        