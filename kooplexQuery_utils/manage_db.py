# This code is a streamlit interface to manage the metadata of the main database
# * Reveal the schema of the database
# * Reveal table and column descriptions
# * Reveal data descriptor and database reference
# * Add examples of questions and corresponding SQL queries
# * Add advices about how to approach a question
# * upload these to the vectorstore

from kooplexQuery_utils.vectorstore import VectorStore
from kooplexQuery.motor import Motor
import pandas as pd
import chromadb
# chromadb.api.client.SharedSystemClient.clear_system_cache()
# from kooplexQuery.db_chat import DBChat
import os
from st_keyup import st_keyup
import streamlit as st

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_schema_to_chat_db(schema_file):
    try:
        raw = schema_file.read()
        try:
            content = raw.decode("utf-8")
        except Exception:
            content = raw.decode("utf-8", errors="replace")
        st.session_state['motor'].db_chat.save_knowledge(reference='schema', content=content)
        st.session_state['db_schema'] = content
        
        st.success("Database schema uploaded to chat database successfully!")
    except Exception as e:
        logger.error(f"Error uploading database schema: {e}")
        st.error("Failed to upload database schema.")

def Database_configuration():
    if st.session_state.get('custom_db_config') is None:
        st.session_state['custom_db_config'] = {
            "hostname": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "mydatabase"),
            "db_user": os.getenv("DB_USER", "myuser"),
            "db_password": os.getenv("DB_PASSWORD", "mypassword"),
            "db_type": os.getenv("DB_TYPE", "postgres"),
            "db_schema": os.getenv("DB_SCHEMA", "distilled"),
            "db_title": os.getenv("DB_TITLE", "My Database")
        }
    st.subheader("Database Information")
    # Open fileds to input database connection details, and update the connection when submitted
    st.file_uploader(label="Upload config file (json)", type="json", key="config_file")
    # Type manually connection details to connect to a database, and update the connection when submitted
    with st.expander("Database Connection Details", expanded=False):
        with st.form("db_info_form"):
            st.text_input(label="Host", value=st.session_state['custom_db_config'].get('hostname'), key="db_host")
            st.text_input(label="User", value=st.session_state['custom_db_config'].get('db_user'), key="db_user")
            st.text_input(label="Password", value="", key="db_password", type="password")
            st.text_input(label="Port", value=st.session_state['custom_db_config'].get('port'), key="db_port")
            st.text_input(label="Database Name", value=st.session_state['custom_db_config'].get('database'), key="db_name")
            st.text_input(label="Schema", value=st.session_state['custom_db_config'].get('schema'), key="schema")
            st.text_input(label="Type", value=st.session_state['custom_db_config'].get('db_type'), key="db_type")
            st.text_input(label="Database Title", value=st.session_state['custom_db_config'].get('db_title'), key="db_title")
            if st.form_submit_button("Connect"):
                st.session_state['custom_db_config'] = {
                    "hostname": st.session_state.db_host,
                    "port": st.session_state.db_port,
                    "database": st.session_state.db_name,
                    "db_user": st.session_state.db_user,
                    "db_password": st.session_state.db_password,
                    "db_type": st.session_state.db_type,
                    "schema": st.session_state.schema,
                    "db_title": st.session_state.db_title
                }
                tmp_config = st.session_state['custom_db_config']
                tmp_config.pop('db_title', None)
                st.session_state['motor'].db_source = st.session_state['motor']._dbtarget_init(tmp_config)
                st.success("Database connection updated!")
    with st.expander("Chat Database Connection Details", expanded=False):
        with st.form("db_chat_form"):
            st.text_input(label="Host", value=st.session_state['custom_db_config'].get('hostname'), key="chat_host")
            st.text_input(label="User", value=st.session_state['custom_db_config'].get('db_user'), key="chat_user")
            st.text_input(label="Password", value="", key="chat_password", type="password")
            st.text_input(label="Port", value=st.session_state['custom_db_config'].get('port'), key="chat_port")
            st.text_input(label="Database", value=st.session_state['custom_db_config'].get('database'), key="chat_database")
            st.text_input(label="Schema", value=st.session_state['custom_db_config'].get('schema'), key="chat_schema")
            if st.form_submit_button("Connect"):
                st.session_state['custom_chatdb_config'] = {
                    "hostname": st.session_state.chat_host,
                    "port": st.session_state.chat_port,
                    "database": st.session_state.chat_database,
                    "db_user": st.session_state.chat_user,
                    "db_password": st.session_state.chat_password,
                    "schema": st.session_state.chat_schema,
                }
                st.session_state['motor'].db_chat = st.session_state['motor']._dbchat_init(st.session_state['custom_chatdb_config'])
                st.success("Database connection updated!")


def main():

    def search_collection(query, collection_name):
        if query:
            collection = st.session_state['vecstore']._init_db(collection_name=collection_name)
            results = collection.similarity_search_with_score(query, k=5)
            for i, (r, score) in enumerate(results):
                with st.expander(f"{i+1} - {r.metadata.get('type')} - Score: {score:.2f} ", expanded=True):
                    st.text(f"{r.page_content}")
                    st.text(f"{'; '.join([f'{k}: {r.metadata[k]}' for k in r.metadata.keys()])}")

        # return rows

    # set page title and layout
    st.set_page_config(page_title="KooplexQuery Database Management", layout="wide")
    
    # Initialize Motor and VectorStore
    if st.session_state.get('motor') is None:
        st.session_state['motor'] = Motor()
        
    if st.session_state.get('vecstore') is None:        
        VECSTORE_PATH = os.getenv("VECSTORE_PATH", "./chroma_vector_db")
        st.session_state['vecstore'] = VectorStore(persist_directory=VECSTORE_PATH)
        # collections = st.session_state['vecstore'].get_collections()
        
    # Get database metadata
    if st.session_state.get('data_descriptor') is None:
        try:
            st.session_state['data_descriptor'] = st.session_state['motor'].db_chat.load_knowledge(reference='data_descriptor')
        except Exception as e:
            logger.error(f"Error loading data descriptor: {e}")
            st.error("No data descriptor found.")   

    # Get database metadata
    if st.session_state.get('instruction') is None:
        try:
            st.session_state['instruction'] = st.session_state['motor'].db_chat.load_knowledge(reference='instruction')
        except Exception as e:
            logger.error(f"Error loading instruction: {e}")
            st.error("No instruction found.")   
            # st.stop()
    if st.session_state.get('db_schema') is None:
        try:
            st.session_state['db_schema'] = st.session_state['motor'].db_chat.load_knowledge(reference='schema')
        except Exception as e:
            logger.error(f"Error loading database schema: {e}")
            st.error("No database schema found.")
            # st.stop()
    if st.session_state.get('database_reference') is None:
        try:
            st.session_state['database_reference'] = st.session_state['motor'].db_chat.load_knowledge(reference='reference')
        except Exception as e:
            logger.error(f"Error loading database reference: {e}")
            st.error("No database reference found.")
            # st.stop()
    if st.session_state.get('table_descriptions') is None:
        try:
            st.session_state['table_descriptions'] = st.session_state['motor'].describe_tables()
        except Exception as e:
            logger.error(f"Error describing tables: {e}")
            st.error("No table descriptions found.")
            # st.stop()
    if st.session_state.get('column_descriptions') is None:
        try:
            st.session_state['column_descriptions'] = st.session_state['motor'].describe_columns()
        except Exception as e:
            logger.error(f"Error describing columns: {e}")
            st.error("No column descriptions found.")
            # st.stop()
    if st.session_state.get('search_query') is None:
        st.session_state['search_query'] = ""
    
    # which tab to show
    if st.session_state.get('current_tab') is None:
        st.session_state['current_tab'] = "Metadata"

    if st.session_state.get('examples') is None:
        try:
            keys, examples = st.session_state['motor'].db_chat.fetch_all_examples()
            table_examples = pd.DataFrame(examples, columns=keys)
            st.session_state['examples'] = table_examples
        except Exception as e:
            logger.error(f"Error fetching examples: {e}")
            st.error("No examples found.")
            # st.stop()

    # Calculate and display statitics of the stored data
    if st.session_state.get('statistics') is None:
        collections_count = {}
        for coll_name in st.session_state['vecstore'].get_collections():
            # logger.info(f"Calculating statistics for collection: {coll_name}")
            coll = st.session_state['vecstore']._init_db(collection_name=coll_name)
            collections_count[coll_name] = len(coll.get()['ids'])
        st.session_state['statistics'] = collections_count

    # This is the container in the main window to show data triggered by buttons in the sidebar    
    disp = st.empty()

    # Sync everything to the vectorstore
    with st.sidebar:
        st.title("KooplexQuery Database Management")
        if st.button("Show Database Schema"):
            st.session_state['current_tab'] = "Schema"
        if st.button("Show Table and Column Descriptions"):
            st.session_state['current_tab'] = "TableColumn"
        if st.button("Show Data Descriptors"):
            st.session_state['current_tab'] = "Metadata"
        if st.button("Show Examples"):
            st.session_state['current_tab'] = "Examples"
        if st.button("Search in the whole VectorStore"):
            st.session_state['current_tab'] = "Search in the whole VectorStore"

        # List Database information
        Database_configuration()

    if st.session_state['current_tab'] == "Schema":
        with disp.container():
            col1, col2 = st.columns(2)
            with col1:
                schema_file = st.file_uploader(label="Upload schema file (ddl)", key="schema_file")
                if schema_file is not None:
                    upload_schema_to_chat_db(schema_file)
                    st.success("Database schema uploaded successfully!")
                    del schema_file
                st.subheader("Original Database Schema")
                st.text(st.session_state.get('db_schema'))
            with col2:
                if st.button("Sync Schema"):
                    st.session_state['vecstore'].load_split_add_text(st.session_state.get('db_schema'), collection_name="schema", split_on="CREATE")
                    st.success("Metadata and examples synced to vectorstore! ")
                st.subheader("Synced Schema")
                schema = st.session_state['vecstore']._init_db(collection_name='schema')
                # Search in the vectorstore for the schema content to verify it was added correctly
                # Search as I type to see the results update in real time
                
                # query = st_keyup("Search in schema collection")
                # result_area = search_schema(query)
                st.write(f"Number of documents in schema collection: {len(schema.get()['ids'])}")
                query = st_keyup("Search in Database Schema", debounce=200)
                if query:
                    search_collection(query, 'schema')
                            
                else:
                    with st.expander("Full schema collection content", expanded=False):
                        first_rows = schema.get()['documents'][:5]
                        st.text("First 5 rows of schema collection:")
                        for row in first_rows:
                            st.text(row)

    elif st.session_state['current_tab'] == "TableColumn":
        with disp.container():
            if st.button("Sync Table and Column Descriptions to VectorStore"):
                for row in st.session_state.get('table_descriptions', ()):
                    # st.session_state['vecstore'].load_split_add_text([f"Table {row[0]}:,  {row[1]}"], collection_name="docs")
                    st.session_state['vecstore'].add_to_docs(metadatas=[{"Table": row[0], 'type': 'table_description'}], texts=[f"{row[1]}"])
                for row in st.session_state.get('column_descriptions', ()):
                    st.session_state['vecstore'].add_to_docs(metadatas=[{"Column": row[0], 'type': 'column_description'}], texts=[f" - ".join(row[1:])])
                st.success("Table and column descriptions synced to vectorstore!")

            docs = st.session_state['vecstore']._init_db(collection_name='docs')
            st.write(f"Number of documents in docs collection: {len(docs.get()['ids'])}")
            query = st_keyup("Search in Table and Column Descriptions", debounce=200)
            if query:
                search_collection(query, 'docs')
                        
            else:
                with st.expander("Full docs collection content", expanded=False):
                    first_rows = docs.get()['documents'][:5]
                    st.text("First 5 rows of docs collection:")
                    for row in first_rows:
                        st.text(row)
                
            col1, col2 = st.columns(2)
            with col1:
                st.header("Table Descriptions")
                with st.expander("Full Table Descriptions", expanded=False):
                    first_rows = st.session_state.get('table_descriptions')[:5]
                    st.text("First 5 rows of table descriptions:")
                    for row in first_rows:
                        st.text(row)
            with col2:
                st.header("Column Descriptions")
                with st.expander("Full Column Descriptions", expanded=False):
                    first_rows = st.session_state.get('column_descriptions')[:5]
                    st.text("First 5 rows of column descriptions:")
                    for row in first_rows:
                        st.text(row)

    elif st.session_state['current_tab'] == "Metadata":
        with disp.container():
            st.write(f"**Statistics**: {st.session_state.get('statistics')}")
            col1, col2 = st.columns(2)
            with col1:
                # Add new instructions
                st.header("Instructions")
                st.markdown(st.session_state.get('instructions'))
                st.text_area("Set prompt  for LLM", value="", key="instructions_input")
                if st.button("Update Instructions"):
                    st.session_state['motor'].db_chat.save_knowledge(reference="instructions",
                                                                     content=st.session_state.instructions_input)
                    st.success("Instructions updated in chat database!")
                    st.session_state['instructions'] = st.session_state['motor'].db_chat.load_knowledge(reference="instructions")
                
                

                # Add new data descriptor
                st.header("Data Descriptor")
                st.text(st.session_state.get('data_descriptor'))
                st.text_area("Explain the data", value="", key="data_descriptor_input")
                if st.button("Update Data Descriptor"):
                    st.session_state['motor'].db_chat.save_knowledge(reference="data_descriptor",
                                                                     content=st.session_state.data_descriptor_input)
                    st.success("Data descriptor updated in chat database!")
                    st.session_state['data_descriptor'] = st.session_state['motor'].db_chat.load_knowledge(reference="data_descriptor")
                
            with col2:
                # Add new data reference
                st.header("Database Reference")
                st.markdown(st.session_state.get('database_reference'))
                st.text_area("Describe the database", value="", key="database_reference_input")
                if st.button("Update Data Reference"):
                    st.session_state['motor'].db_chat.save_knowledge(reference="data_reference",
                                                                     content=st.session_state.database_reference_input)
                    st.success("Data reference updated in chat database!")
                    st.session_state['database_reference'] = st.session_state['motor'].db_chat.load_knowledge(reference="data_reference")
                
        
    elif st.session_state['current_tab'] == "Examples":
        with disp.container():
            examples = st.session_state.get('examples')
            if st.button("Sync Examples to VectorStore"):
                
                for row in examples.itertuples(index=False):
                    if row.public:  # Only add public examples to the vectorstore
                        st.session_state['vecstore'].add_to_examples({'question': row.question_content, 'sql': row.sql})

                st.success("All public/validated examples were synced to the vectorstore!")
                
            vs_examples = st.session_state['vecstore']._init_db(collection_name="examples")
            st.write(f" {len(vs_examples.get()['ids'])} examples are in the vectorstore...")
            # Add new example
            # question_content = st.text_input("Question")
            # sql_query = st.text_input("SQL Query")
            # if st.button("Add new example"):

            #         # Here you would implement the logic to add the new example to your database and vectorstore
            #         st.success("New example added!")
            #         # Optionally, you could also update the session state to reflect the new example
            #         # Add examples of questions and corresponding SQL queries
            #         st.session_state['motor'].db_chat.save_query(session_id="example_session",
            #             question_type = 'train', 
            #             question_content=question_content, 
            #             sql=sql_query)
            #         st.success("Example added to vectorstore!")

            st.dataframe(examples)

    elif st.session_state['current_tab'] == "Search in the whole VectorStore":
        with disp.container():
                st.multiselect("Select collection to search", options=st.session_state['vecstore'].get_collections(), default=st.session_state['vecstore'].get_collections() if st.session_state['vecstore'].get_collections() else None, key="search_collection")
                # st.text_input("Enter a search query", key="search_query")
                query = st_keyup("Search in all collections", debounce=200)
                if query:
                    for coll_name in st.session_state.search_collection:
                        st.subheader(f"Results from collection: {coll_name}")
                        search_collection(query, coll_name)

if __name__ == "__main__":
    main()