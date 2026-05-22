# from projects.text2sql.david.kooplexQuery.kooplexQuery_utils.misc import upload_schema_to_chat_db

from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
from streamlit_extras.chart_container import *
import streamlit.components.v1 as components
from streamlit_elements import html
import streamlit.web.cli as stcli

import logging
import time
from kooplexQuery.motor import LLM_Model, Motor, add_llm_model, delete_llm_model, list_llm_models
from kooplexQuery.utils.misc import *
from kooplexQuery.utils.sync_manager import *
from kooplexQuery.utils.plot_utils import *
import asyncio
import uuid
import sys
import os
from pathlib import Path
import pandas as pd

logging.basicConfig(
    filename='/tmp//app.log', 
    level=logging.DEBUG
)
# Suppress verbose file-watcher debug noise from Streamlit/watchdog.
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.debug("START")

def prompt():
    # Set some session state defaults
    st.session_state.setdefault('sql_submit', None)
    st.session_state.setdefault('sql_fix', None)
    st.session_state.setdefault('sql_saved', None)
    st.session_state.setdefault('plot_instruction', None)
    st.session_state.setdefault('save_req', False)
    st.session_state.setdefault('rerun', False)
    st.session_state.setdefault('session_init_failed', False)

    st.session_state.setdefault('df', None)


    ###
    username="fakeuser"
    email="fake@em.ail"
    #####
    st.set_page_config(layout="wide")
    # Inject CSS globally
    st.html("""
    <style>
    div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
        width: 80vw;
        height: None;
    }
    </style>
    """)

    # Main containers
    example_container = st.empty()
    histo_container = st.empty()
    control_container = st.empty()

    # This is the container in the main window to show data triggered by buttons in the sidebar    
    disp = st.empty()

    # @st.dialog("Database Configuration")
    # def database_configuration():
    #     dsm = DatabaseServerManager()
    #     dsm.Database_configuration()    

    @st.dialog("Set environment variables")
    def config_env_dialog():
        st.markdown("### Set environment variables for database connection")
        st.markdown("These variables will be saved into the `config.env` file and loaded on the next startup to initialize the Motor. After setting these, please restart the app.")
        env_vars = {
            "OPENAI_API_KEY": {"value": "OpenAI API key", "type": "password"},
            "OLLAMA_HOST": {"value": "Ollama host", "type": "default"},
            "OLLAMA_PORT": {"value": "Ollama port", "type": "default"},
            "OLLAMA_MODEL": {"value": "Ollama model", "type": "default"},
            "ANTHROPIC_API_KEY": {"value": "Anthropic API key", "type": "password"},
        }
        #if config.env file exists, load existing values as defaults
        export_env_variables()
        for var in env_vars.keys():
            env_vars[var]['value'] = os.getenv(var, "")
        new_values = {}
        for var, field in env_vars.items():
            new_values[var] = st.text_input(var, value=field["value"], type=field["type"])
        
        if st.button("Save environment variables"):
            save_env_variables(new_values)
            export_env_variables()
            st.success("Environment variables saved! Please restart the app for changes to take effect.")

    def export_env_variables():
        config_path = "./config.env"
        if os.path.exists(config_path):
            load_dotenv(config_path)
        else:
            st.warning("No config.env file found to export.")

    def save_env_variables(new_values):
        st.session_state['environment_variables_set'] = False
        # config_path = Path(__file__).parent.parent / "config.env"
        config_path =  "./config.env"
        with open(config_path, "w") as f:
            for var, value in new_values.items():
                f.write(f"{var}={value}\n")
        logger.info(f"Environment variables saved to {config_path}")

    # Pop up dialogs
    @st.dialog("DB browser")
    def show_schema_browser():
        import pandas
        st.html("<span class='big-dialog'></span>")
        table_description=dict(st.session_state.motor.describe_tables())
        column_description=pandas.DataFrame(st.session_state.motor.describe_columns(), columns=['table', 'column', 'description', 'type'])
        selected_tables = st.multiselect("Select Tables", table_description.keys())
        if selected_tables:
            for t in selected_tables:
                with st.expander(f"Table {t}"):
                    st.write(table_description[t])
                with st.expander(f"Column Descriptions for {t}"):
                    st.write(column_description[column_description['table']==t].drop(columns=['table']))
                with st.expander(f"Example records from {t}"):
                    sql=f"SELECT * FROM {t} ORDER BY RANDOM() LIMIT 2"
                    st.write(st.session_state.motor.db_source.query_to_df(sql))


    # Form where user may modify the SQL query
    def form_sql(sql, fix_error=None, last_record=False):
        uid=uuid.uuid4()
        with st.form(f'f-{uid}'):
            st.text_area(
                "You may rewrite the query proposal or run it directly:", 
                value=sql.strip(),
                key=f"ns-{uid}",
                help="If you are familiar with SQL syntax you may modify and fine tune the query and rerun")
            def _t():
                st.session_state.sql_submit=getattr(st.session_state, f'ns-{uid}')
            st.form_submit_button("🛢️ Run the query", on_click=_t)
            if fix_error:
                if last_record and st.session_state.autocorrect and st.session_state.sql_fix is None:
                    st.session_state.sql_fix=(sql, fix_error)
                    st.rerun()
                def _t():
                    st.session_state.sql_fix=(sql, fix_error)
                st.form_submit_button("🔨 Let LLM fix", on_click=_t)


    # Form where user may modify plot instructions
    def form_plot(plot_instructions="Plot data"):
        uid=uuid.uuid4()
        with st.form(f'f-{uid}'):
            st.text_area(
                "Enter your instructions for plotting the data",
                value=plot_instructions.strip(),
                key=f"pi-{uid}",
                help="Provide instructions on how you want the data to be plotted. For example, 'Plot the histogram of temperature and pH'."
            )
            def _t():
                st.session_state.plot_instruction=getattr(st.session_state, f'pi-{uid}')
            st.form_submit_button("📊 Generate Plot", on_click=_t)


    # Parse and write LLM response
    def render_sql_parsed(chunks):
        for chunk in chunks:
            if chunk.type=='txt':
                st.write(chunk.content)
            elif chunk.type=='sql':
                sql=chunk.content
                tab1, tab2 = st.tabs(["🛢️ Sql", "🦖 Rewrite code"])
                with tab1:
                    with st.form(str(uuid.uuid4())):
                        def _q(q):
                            st.session_state.sql_submit=q
                        st.code(sql, language="sql", wrap_lines=True)
                        st.form_submit_button(on_click=_q, args=(sql,))
                with tab2:
                    form_sql(sql)
            else:
                st.error("Unhandled chunk")
                st.write(chunk)


    # Print dataset
    def render_df(sql, df):
        tab1, tab2, tab3 = st.tabs(["📋 Dataset", "🚀 Plot", "🦖 Rewrite SQL"])
        with tab1:
            st.dataframe(df)
        with tab2:
            form_plot()
        with tab3:
            form_sql(sql)

    # Show plot
    def render_plot(instructions, answer, pyplot):
        fig=answer['content']
        tab1, tab2, tab3 = st.tabs(["📈 Plot", "🧠 Code", "🚀 Replot"])
        with tab1:
            if pyplot:
                st.pyplot(fig)
            else:
                st.plotly_chart(fig, key=str(uuid.uuid4()))
        with tab2:
            st.code(answer["code"])
        with tab3:
            form_plot(instructions)

    # Present error message
    def render_error(answer, last_record):
        _tabs=["💥 Error"]
        error=answer["content"]
        if "code" in answer:
            _tabs.append("🧠 Code")
        if "query" in answer:
            _tabs.append("🛠️ Fix SQL code")
        tabs=st.tabs(_tabs)
        with tabs[0]:
            e=getattr(error, 'orig', error)
            st.error(e)
        if "code" in answer:
            with tabs[1]:
                st.code(answer["code"])
        if "query" in answer:
            with tabs[-1]:
                form_sql(answer['query'], fix_error=error, last_record=last_record)

    # Display chat history
    def history():
        with histo_container.container():
            for r in st.session_state.motor.chat_history:
                prompt=r['question']
                response=r['answer']
                response_meta=r['answer_meta']
                response_type=response_meta.get('type')
                last_pair=r['is_last']
                label = f"{prompt[:80]}{'...' if len(prompt) > 80 else ''}"
                with st.expander(label=label, expanded=last_pair):
                    with st.chat_message("user"):
                        if r['question_meta'].get('type')=='submit_sql':
                            st.code(prompt, language="sql", wrap_lines=True)
                        else:
                            st.write(prompt)
                    with st.chat_message("assistant"):
                        if response_type=="dataframe":
                            render_df(prompt, response_meta['dataframe'])
                        elif response_type in ["pyplot", "plotly_chart"]:
                            render_plot(prompt, response_meta, response_type=="pyplot")
                        elif response_type=="error":
                            render_error(response_meta, last_pair)
                        else:
                            render_sql_parsed(response_meta.get('parsed'))
                    if dt:=response_meta.get('duration'):
                        st.info(f"duration: {dt} s")
                    if last_pair:
                        if st.button('Delete'):
                            st.session_state.motor.pop()
                            st.rerun()
    #        if len(chat_history)%1 == 1:
    #            user_msg=chat_history[-1]
    #            st.chat_message("user").write(user_msg["content"])

    # If environmental variables were not yet read then load it with dotenv
    if 'environment_variables_set' not in st.session_state or    st.session_state['environment_variables_set'] == False:
        from dotenv import load_dotenv
        load_dotenv(os.environ.get('KOOPLEX_CONFIG_ENV_PATH', 'config.env'))

    # If motor startup failed, expose DB settings and stop the rest of the app flow.
    if st.session_state.get('db_config_set') is None:
        st.warning("Please configure database settings and press Connect. If session creation fails, open Database settings and try again.")
        # st.stop()    

    # If connect was not yet pressed show the configurator
    if st.session_state.get('db_config_set') is None:
        st.session_state['interface'] = "dbconfig"
        with st.sidebar:
            st.title("KooplexQuery")
            #st.error("Backend initialization failed.")
            if st.session_state.get('motor_init_error'):
                st.caption(st.session_state['motor_init_error'])
            # if st.button("Database settings", width='stretch'):
        # with disp.container():
    
    # Set the first db config to show
    if st.session_state.get('selected_db_index') is None:
        st.session_state['selected_db_index'] = 0

    if st.session_state.get('interface') == "dbconfig":
        dsm = DatabaseServerManager()
        dsm.Database_configuration()    
        

    # Initialize the backend motor
    if st.session_state.get('motor') is None:
        logger.info(f"Initialize motor")
        try:
            # st.session_state.motor = Motor(table_name_filter='viralprimer_%')
            st.session_state.motor = Motor(table_name_filter='%')
            st.session_state['motor_init_error'] = None
        except Exception as e:
            logger.error(f"Motor initialization failed: {e}")
            st.session_state['motor_init_error'] = str(e)
            st.session_state['motor'] = None

    # Init view
    # if st.session_state.get('interface') is None:
        # st.session_state.interface = 'chat'

    # Convenient function to clear parts of the session state
    def clear_session_keys(*keys):
        for key in keys:
            if key in st.session_state:
                st.session_state[key] = None

    # A helper to setup a new session
    def _newsession():
        try:
            st.session_state['session']=st.session_state.motor.new_session(username=username, email=email) #TODO: label, referenced_session
            st.session_state['session_init_failed'] = False
        except Exception as e:
            logger.error(f"Error initializing a new session: {e}")
            st.error("Error initializing a new session. Please check the backend connection and try again.")
            st.session_state['session'] = None
            st.session_state['session_init_failed'] = True
            return
        clear_session_keys('sql_submit', 'save_success') #FIXME
        st.session_state.selected_question = {'question_id': None, 'question': None, 'sql': None, 'score': None, 'type': None, 'public': None}    
        st.rerun()

    

    # Setup a new session on the first run, or recover if a partially initialized
    # Motor object exists without a persisted session_id.
    needs_session = (
        'session' not in st.session_state
        or st.session_state.get('session') is None
        or not hasattr(st.session_state.motor, 'session_id')
    )
    if needs_session and not st.session_state.get('session_init_failed'):
        #TODO ask for a session label
        #TODO elaborate meta and be it json dump
        logger.info(f"Creating new session")
        _newsession()

    def format_model_option(model: LLM_Model) -> str:
        return f"{model.model_name} - {model.provider}"

    def model_ref(model: LLM_Model) -> tuple[str, str]:
        return (model.provider, model.model_name)

    def selected_model_index(models: list[LLM_Model]) -> int:
        selected_ref = st.session_state.get('selected_model_ref')
        if selected_ref is not None:
            for i, model in enumerate(models):
                if model_ref(model) == selected_ref:
                    return i
        st.session_state['selected_model_ref'] = model_ref(models[0])
        return 0


# From managedb

    def search_collection(query, collection_name):
        if query:
            collection = st.session_state['vecstore']._init_db(collection_name=collection_name)
            results = collection.similarity_search_with_score(query, k=5)
            for i, (r, score) in enumerate(results):
                with st.expander(f"{i+1} - {r.metadata.get('type')} - Score: {score:.2f} ", expanded=True):
                    st.text(f"{r.page_content}")
                    st.text(f"{'; '.join([f'{k}: {r.metadata[k]}' for k in r.metadata.keys()])}")


    if st.session_state.get('interface') != 'dbconfig':
        # Init
        # Initialize Motor and VectorStore
        if st.session_state.get('motor') is None:
            st.session_state['motor'] = Motor()
            st.session_state['motor'].db_chat = st.session_state['motor']._dbchat_init()
            st.session_state['motor'].db_source = st.session_state['motor']._dbtarget_init()

        # if 'database_server'  in st.session_state:
            # st.write(f"Database server config: {st.session_state.database_server}")
            # st.write(st.session_state.get('vecstore') , st.session_state.database_server.get('title'), (st.session_state.get('vecstore') is None and st.session_state.database_server.get('title')) )
        if st.session_state.get('vecstore') is None and 'database_server' in st.session_state:   
            VECSTORE_PATH = os.getenv("VECSTORE_PATH", st.session_state.database_server.get('title', "kooplexquery_db"))
            try:
                st.session_state['vecstore'] = VectorStore(persist_directory=VECSTORE_PATH )
                # Connect vectorstore to Motor's sync manager for automatic synchronization
                st.session_state['motor'].vectorstore = st.session_state['vecstore']
                logger.info("VectorStore connected to sync manager for automatic synchronization")
            except Exception as e:
                logger.error(f"Error initializing vector store: {e}")
                st.error("Error initializing vector store. Please check the configuration and try again.")
                st.session_state['vecstore'] = None
        
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
        if st.session_state.get('meta_schema') is None:
            try:
                st.session_state['meta_schema'] = st.session_state['motor'].db_chat.load_knowledge(reference='schema')
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
    if st.session_state.get('statistics') is None and st.session_state.get('vecstore') is not None:
        collections_count = {}
        for coll_name in st.session_state['vecstore'].get_collections():
            # logger.info(f"Calculating statistics for collection: {coll_name}")
            coll = st.session_state['vecstore']._init_db(collection_name=coll_name)
            collections_count[coll_name] = len(coll.get()['ids'])
        st.session_state['statistics'] = collections_count



    # Page render logic
    with st.sidebar:
        # Two buttons that switches between the chat/validator and the metadata manager view
        if st.button("Metadata manager", width='stretch'):
            st.session_state.interface = "managedb"
            _newsession()
            st.rerun()
        if st.button("Chat", width='stretch'):
            st.session_state.interface = "chat"
            _newsession()
            st.rerun()
        if st.button("Validator", width='stretch'):
            st.session_state.interface = "validator"
            _newsession()
            st.rerun()

        if st.session_state.interface == 'managedb':
            st.title("KooplexQuery Database Management")
            st.subheader(st.session_state['database_server'].get('title', "Database"))
            if st.button("Documentation"):
                st.session_state['current_tab'] = "Documentation"
            if st.button("Table and Column Descriptions"):
                st.session_state['current_tab'] = "TableColumn"
            if st.button("Examples"):
                st.session_state['current_tab'] = "Examples"
            if st.button("Search in the whole VectorStore"):
                st.session_state['current_tab'] = "Search in the whole VectorStore"
        elif st.session_state.interface == 'chat' or st.session_state.interface == 'validator' :
            page_title = os.getenv("TITLE", "KooplexQuery - Text2SQL with LLMs")
            st.title(st.session_state['database_server'].get('title', page_title))
            st.subheader(page_title)
            
            if st.button("New Session", width='stretch', disabled=st.session_state.motor.is_new_session):
                _newsession()
            st.toggle("Autocorrect", value=False, key="autocorrect", on_change=None, args=None, kwargs=None)
            available_models = list_llm_models()
            model_index = selected_model_index(available_models)
            selected_model = st.sidebar.selectbox(
                "Select model for Text2SQL",
                available_models,
                index=model_index,
                format_func=format_model_option,
            )
            st.session_state.selected_model_ref = model_ref(selected_model)
            st.session_state.selected_model = selected_model
            st.session_state.current_model = selected_model.model_name

            with st.form("add_llm_model_form"):
                new_model_name = st.text_input("New LLM model", key="new_llm_model_name")
                new_model_provider = st.text_input("New LLM model provider", key="new_llm_model_provider")
                add_model_requested = st.form_submit_button("Add model")
            if add_model_requested:
                try:
                    added_model = add_llm_model(new_model_name, new_model_provider)
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    st.session_state.selected_model_ref = model_ref(added_model)
                    st.session_state.current_model = added_model.model_name
                    st.rerun()

            if st.button("Delete selected model", width='stretch'):
                deleted = delete_llm_model(selected_model.model_name, selected_model.provider)
                if deleted:
                    st.session_state.selected_model_ref = None
                    st.success(f"Deleted model: {selected_model.model_name} ({selected_model.provider})")
                    st.rerun()
                else:
                    st.warning("Selected model was not found in the model registry.")

            if st.button("Save Accurate Query", width='stretch', disabled=not st.session_state.motor.can_prepare_save):
                st.session_state.save_req=True
                st.rerun()

            if st.session_state.interface == 'validator':
                if st.button("Delete", width='stretch', disabled=not st.session_state.selected_question['question']):
                    st.session_state.motor.db_chat.delete_row(int(st.session_state.selected_question['question_id']))
                    _newsession()

                if st.button("Validate", width='stretch', disabled=not st.session_state.selected_question['question']):
                    st.session_state.motor.db_chat.validate_question(int(st.session_state.selected_question['question_id']))
                    _newsession()

        st.markdown("---")

        # List Database information
        if st.button("Database settings", width='stretch'):
            st.session_state['interface'] = "dbconfig"
            # with disp.container():
            #     dsm = DatabaseServerManager()
            #     dsm.Database_configuration()    
            st.rerun()



        if st.button("Schema browser", width='stretch'):
            show_schema_browser()
        
        # A dialog to save environment variables into the config.env file that is going to be loaded on the next startup to initialize the Motor
        if st.button("Set environment variables", width='stretch'):
            st.session_state['show_env_dialog'] = True
            config_env_dialog()
            
        if st.button("Reset Vectorstore", width='stretch'):
            # Clear the vectorstore collection and resync with the latest database metadata
            for collection in st.session_state['vecstore'].get_collections():
                st.session_state['vecstore']._init_db(collection_name=collection).delete_collection()
            try:
                # Keep sync manager bound to the latest db_chat and vectorstore.
                st.session_state['motor']._ensure_sync_manager()
                st.session_state['motor'].vectorstore = st.session_state['vecstore']

                

                # Also sync table/column descriptions from the active source DB.
                table_descriptions = st.session_state['motor'].describe_tables() or ()
                column_descriptions = st.session_state['motor'].describe_columns() or ()
                for row in table_descriptions:
                    st.session_state['vecstore'].add_to_docs(
                        metadatas=[{"Table": row[0], 'type': 'table_description'}],
                        texts=[f"{row[1]}"]
                    )
                for row in column_descriptions:
                    st.session_state['vecstore'].add_to_docs(
                        metadatas=[{"Column": row[0], 'type': 'column_description'}],
                        texts=[f" - ".join(row[1:])]
                    )

                sync_results = st.session_state['motor'].sync_manager.resync_all()

                # Force stats to recalculate from the freshly synced vectorstore.
                st.session_state['statistics'] = None
                st.session_state['pending_vectorstore_sync'] = False
                logger.info(f"Automatic vectorstore sync completed after connect: {sync_results}")
            except Exception as e:
                logger.error(f"Automatic vectorstore sync after connect failed: {e}")
                st.warning("Connected, but automatic vectorstore sync failed. You can still sync manually from Metadata manager.")


    # The page body
    if st.session_state.motor is not None:
        if st.session_state.interface == 'chat':
            def show_examples(n_examples=3):
                    if st.session_state.motor.is_new_session:
                        # Check if there are examples available and show them only if there are and we are at the start of the session
                        if st.session_state.motor.fetch_examples(1):
                            with example_container.container():
                                with st.form("example_form"):
                                    st.markdown("##### _Example questions_")
                                    cols = st.columns(n_examples)
                                    def _s(prompt, sql):
                                        st.session_state.motor.select_example(prompt, sql)
                                        st.session_state.rerun=True
                            
                                    for i, (_prompt, _sql) in enumerate(st.session_state.motor.fetch_examples(n_examples)):
                                        cols[i].form_submit_button(_prompt, type="secondary", on_click=_s, args=[_prompt, _sql])
                        else:
                            with example_container.container():
                                    st.markdown("##### _There are no example questions yet_")

            

        elif st.session_state.interface == 'validator':
            def show_examples():
                with example_container.container():
                    # st.write("### Examples (select one to validate)")
                    if st.session_state.motor.is_new_session:
                            keys, data = st.session_state.motor.db_chat.fetch_all_examples()
                            st.header(f"Select an example from the database to validate.")
                            st.write(f"{len(data)} examples available.")
                            st.session_state.df = pd.DataFrame(columns=keys, data=data)
                            #st.session_state.df = df[['question_id', 'question_content', 'sql', 'type', 'public', 'score']]
                            st.session_state.df.sort_values(by=['public','score', 'question_content' ], ascending=True, inplace=True)
                            # Add a data editor with row selection
                            event = st.dataframe(
                                st.session_state.df,
                                width='stretch',
                                hide_index=True,
                                on_select="rerun",
                                
                            )

                            # If a row is selected, continue with its value in a certain column
                            # If a row is selected, continue with its value in a certain column
                            selected_rows = event.selection.rows
                            if selected_rows:
                                selected_idx = selected_rows[0]
                                selected_row = st.session_state.df.iloc[selected_idx]
                                st.session_state.selected_question = {
                                    'question_id': selected_row['question_id'],
                                    'question' : selected_row['question_content'],
                                    'sql' : selected_row['sql'],
                                    'score' : selected_row['score'],
                                    'type' : selected_row['type'],
                                    'public' : selected_row['public'],
                                }
                                # For example, continue with the value in the 'content' column
                                st.info(f"Continuing with: {st.session_state.selected_question }")
                                st.session_state.motor.select_example(st.session_state.selected_question['question'],
                                                                    st.session_state.selected_question['sql'])
                    else:
                        st.write("There are no examples yet")
            
            

    if st.session_state.interface == 'chat' or st.session_state.interface == 'validator':
        show_examples()
        if st.session_state.rerun:
            st.session_state.rerun=False
            st.rerun()
        history()

        # Save requested
        if st.session_state.save_req:
            with control_container.container():
                st.markdown("💡 Note: _An expert will validate this relation later._")
                with st.spinner("Digesting conversation..."):
                    st.write_stream(st.session_state.motor.prepare_save(st.session_state.selected_model))
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("💾 Save", disabled=not st.session_state.motor.can_save):
                            logger.info(f"Query saved: {st.session_state.sql_saved}")

                            st.session_state.sql_saved=st.session_state.motor.sql
                            st.session_state.motor.save_query()
                            st.session_state.save_req=False
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel"):
                            st.session_state.save_req=False
                            st.rerun()

        # Show success
        if st.session_state.sql_saved:
            st.session_state.sql_saved=None
            st.success("Saved query")


        # Submit SQL
        if st.session_state.sql_submit:
            with st.spinner("✅ Data retrieval..."):
                st.session_state.motor.execute(st.session_state.sql_submit)
                st.session_state.sql_submit=None
                st.rerun()

        # Fix SQL
        if st.session_state.sql_fix:
            with control_container.container():
                with st.spinner("⏳ Thinking...."):
                    with st.chat_message("assistant"):
                        st.write_stream(st.session_state.motor.correct_error(st.session_state.sql_fix[1], st.session_state.selected_model))
                st.session_state.sql_fix=None
                st.rerun()

        # Generate plot
        if st.session_state.plot_instruction:
            with st.spinner("✅ Generating plot"):
                with st.chat_message("assistant"):
                    st.write_stream(st.session_state.motor.plot(st.session_state.plot_instruction, st.session_state.selected_model))
                    st.session_state.plot_instruction=None
                    st.rerun()


        # Handle prompt input
        if "awaiting_response" not in st.session_state:
            st.session_state.awaiting_response = False
        if "latest_user_input" not in st.session_state:
            st.session_state.latest_user_input = None

        # Show input only if not currently processing
        if not st.session_state.awaiting_response:
            user_input = st.chat_input("Enter a prompt", key="user_input")
            if user_input:
                st.session_state.latest_user_input = user_input
                st.session_state.awaiting_response = True
                st.rerun()

        # Show response if awaiting
        if st.session_state.awaiting_response and st.session_state.latest_user_input:
            with st.chat_message("user"):
                st.write(st.session_state.latest_user_input)

            with st.spinner("⏳ Thinking..."):
                with st.chat_message("assistant"):
                    st.write_stream(st.session_state.motor.chat(st.session_state.latest_user_input, st.session_state.selected_model))

            # Reset state and allow new input
            st.session_state.awaiting_response = False
            st.session_state.latest_user_input = None
            st.rerun()

    # Manage db button logics
    if st.session_state.interface == 'managedb':
        if st.session_state['current_tab'] == "Documentation":
            with disp.container():
                try:
                    keys, knowledge_rows = st.session_state['motor'].db_chat.fetch_all_knowledge()
                    knowledge_df = pd.DataFrame(knowledge_rows, columns=keys)
                except Exception as e:
                    logger.error(f"Error fetching knowledge rows: {e}")
                    st.error("Could not load rows from the knowledge table.")
                    knowledge_df = pd.DataFrame(columns=["id", "reference", "content"])

                st.subheader("Knowledge Table")

                if knowledge_df.empty:
                    st.info("No rows found in the knowledge table.")
                else:
                    # st.write(f"{len(knowledge_df)} knowledge rows found.")
                    
                    # if st.button("🔄 Sync Knowledge to VectorStore"):
                    #     try:
                    #         for _, row in knowledge_df.iterrows():
                    #             st.session_state['vecstore'].add_to_docs(
                    #                 metadatas=[{"Reference": row['reference'], 'type': 'knowledge', 'id': row['id']}],
                    #                 texts=[row['content']]
                    #             )
                    #         st.success(f"Synced {len(knowledge_df)} knowledge entries to vectorstore!")
                    #     except Exception as e:
                    #         logger.error(f"Error syncing knowledge to vectorstore: {e}")
                    #         st.error(f"Failed to sync knowledge: {e}")
                    selected_index = st.selectbox(
                        "Select a knowledge row",
                        options=knowledge_df.index.tolist(),
                        format_func=lambda idx: f"{knowledge_df.loc[idx, 'reference']} (id={knowledge_df.loc[idx, 'id']})",
                    )
                    selected_row = knowledge_df.loc[selected_index]

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(
                            knowledge_df[["id", "reference"]],
                            width='stretch',
                            hide_index=True,
                        )
                    with col2:
                        st.markdown(f"**ID:** {selected_row['id']}")
                        st.markdown(f"**Reference:** {selected_row['reference']}")
                        st.markdown("**Content:**")
                        edited_content = st.text_area(
                            "Knowledge content",
                            value=selected_row['content'] or "",
                            height=400,
                            disabled=False,
                            label_visibility="collapsed",
                            key=f"content_edit_{selected_index}",
                        )
                        col_update, col_delete = st.columns(2)
                        with col_update:
                            if st.button("💾 Update Row", key=f"update_{selected_index}"):
                                try:
                                    st.session_state['motor'].db_chat.save_knowledge(
                                        reference=selected_row['reference'],
                                        content=edited_content
                                    )
                                    st.success(f"Updated knowledge row: {selected_row['reference']}")
                                    st.rerun()
                                except Exception as e:
                                    logger.error(f"Error updating knowledge row: {e}")
                                    st.error(f"Failed to update row: {e}")
                        with col_delete:
                            if st.button("🗑️ Delete Row", key=f"delete_{selected_index}"):
                                try:
                                    st.session_state['motor'].db_chat.engine.execute(
                                        text(f"DELETE FROM {st.session_state['motor'].db_chat.schema}.knowledge WHERE id = :id"),
                                        {'id': selected_row['id']}
                                    )
                                    st.session_state['motor'].db_chat.engine.commit()
                                    st.success(f"Deleted knowledge row: {selected_row['reference']}")
                                    st.rerun()
                                except Exception as e:
                                    logger.error(f"Error deleting knowledge row: {e}")
                                    st.error(f"Failed to delete row: {e}")

                st.markdown("---")
                st.subheader("Add New Knowledge Entry")
                with st.form("add_knowledge_form"):
                    new_reference = st.text_input("Reference", placeholder="e.g., schema, instruction, data_descriptor")
                    new_content = st.text_area("Content", height=300, placeholder="Enter the knowledge content here")
                    submitted = st.form_submit_button("✨ Add New Row")
                    
                    if submitted:
                        if not new_reference or not new_content:
                            st.error("Both Reference and Content are required")
                        else:
                            try:
                                st.session_state['motor'].db_chat.save_knowledge(
                                    reference=new_reference,
                                    content=new_content
                                )
                                st.success(f"Added new knowledge entry: {new_reference}")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Error adding knowledge row: {e}")
                                st.error(f"Failed to add row: {e}")
        elif st.session_state['current_tab'] == "TableColumn":
            with disp.container():
                st.header("Table and Column Descriptions")
                st.write("Descriptions of tables and columns extracted from the connected database. You can use this information to understand the database structure and also as a source to sync with the vectorstore for better question understanding by the LLM.")

                # if st.session_state.get('vecstore') is None:
                #     st.warning("Vector store is not available. Please check configuration and restart the app.")
                # else:
                    # if st.button("Sync Table and Column Descriptions to VectorStore"):
                    #     for row in st.session_state.get('table_descriptions', ()):
                    #         # st.session_state['vecstore'].load_split_add_text([f"Table {row[0]}:,  {row[1]}"], collection_name="docs")
                    #         table_name = str(row[0]) if len(row) > 0 else ""
                    #         table_desc = table_name if len(row) < 2 or row[1] is None else str(row[1])
                    #         st.session_state['vecstore'].add_to_docs(
                    #             metadatas=[{"Table": table_name, 'type': 'table_description'}],
                    #             texts=[table_desc]
                    #         )
                        # for row in st.session_state.get('column_descriptions', ()):
                        #     column_name = str(row[1]) if len(row) > 1 else (str(row[0]) if row else "")
                        #     text_parts = [str(value) for value in row if value is not None]
                        #     st.session_state['vecstore'].add_to_docs(
                        #         metadatas=[{"Column": column_name, 'type': 'column_description'}],
                        #         texts=[" - ".join(text_parts)]
                        #     )
                        # st.success("Table and column descriptions synced to vectorstore!")

                    # docs = st.session_state['vecstore']._init_db(collection_name='docs')
                    # # st.write(f"Number of documents in docs collection: {len(docs.get()['ids'])}")
                    # with st.form("table_column_search_form"):
                    #     query = st.text_input("Search in Table and Column Descriptions", key="table_column_search_query")
                    #     search_docs = st.form_submit_button("Search")
                    # if search_docs and query:
                    #     search_collection(query, 'docs')
                                
                    # else:
                    #     with st.expander("Full docs collection content", expanded=False):
                    #         first_rows = docs.get()['documents'][:5]
                    #         st.text("First 5 rows of docs collection:")
                    #         for row in first_rows:
                    #             st.text(row)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Tables")
                    with st.expander("Full Table Descriptions", expanded=False):
                        first_rows = st.session_state.get('table_descriptions')[:5]
                        st.text("First 5 rows of table descriptions:")
                        for row in first_rows:
                            st.text(row)
                with col2:
                    st.subheader("Columns")
                    with st.expander("Full Column Descriptions", expanded=False):
                        first_rows = st.session_state.get('column_descriptions')[:5]
                        st.text("First 5 rows of column descriptions:")
                        for row in first_rows:
                            st.text(row)

       
        elif st.session_state['current_tab'] == "Examples":
            with disp.container():
                st.header("Examples")
                st.write("Examples of questions and corresponding SQL queries stored in the database. You can add new examples to the database which can then be used to guide the LLM in understanding how to answer questions based on the connected database. You can also sync the examples with the vectorstore to make them available for retrieval when relevant to the user's question.")
                st.markdown("*Use the **Validator** menu to validate the examples and make them public!*")
                examples = st.session_state.get('examples')
                if st.session_state.get('vecstore') is None:
                    st.warning("Vector store is not available. Please check configuration and restart the app.")
                else:
                    # if st.button("Sync Examples to VectorStore"):
                    #     # Use sync manager for batch sync of all public examples
                    #     synced_count = st.session_state['motor'].sync_manager.batch_sync_examples()
                    #     st.success(f"All {synced_count} public/validated examples were synced to the vectorstore!")
                        
                    # vs_examples = st.session_state['vecstore']._init_db(collection_name="examples")
                    # st.write(f" {len(vs_examples.get()['ids'])} examples are in the vectorstore...")
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
                st.header("Search in the whole VectorStore")
                st.write("To aid the AI assistant in understanding your database and providing accurate answers, the metadata is stored in a vectorstore")
                st.write("You can search through the vectorstore to see how it finds relevant information for your questions.")
                st.markdown(f"Vectorstore contains {st.session_state['statistics']} entries")
                if st.session_state.get('vecstore') is None:
                    st.warning("Vector store is not available. Please check configuration and restart the app.")
                else:
                    st.multiselect("Select collections to search in", options=st.session_state['vecstore'].get_collections(), default=st.session_state['vecstore'].get_collections() if st.session_state['vecstore'].get_collections() else None, key="search_collection")
                    with st.form("all_collections_search_form"):
                        query = st.text_input("Search in all collections", key="all_collections_search_query")
                        search_all = st.form_submit_button("Search")
                    if search_all and query:
                        for coll_name in st.session_state.search_collection:
                            st.subheader(f"Results from collection: {coll_name}")
                            search_collection(query, coll_name)

    logger.debug("END ----")
    #logger.info(f"Session {st.session_state.motor.chat_history} - End of prompt function")

def resolve_path(path=None):
    here = Path(__file__).resolve()
    if path is None:
        return str(here)
    return str((here.parent / path).resolve())


def main():
    import kooplexQuery.prompt as prompt_module
    script_path = Path(prompt_module.__file__).resolve()

    # 2. Define the Streamlit arguments
    # Note: baseUrlPath should NOT have leading/trailing slashes

    # Import environment variables from .env file if it exists
    import os
    _ge = lambda x,d: os.getenv(x, d)

    from dotenv import load_dotenv
    if not load_dotenv():
        load_dotenv(os.environ.get('KOOPLEX_CONFIG_ENV_PATH', 'config.env'))

    # Check environment variables for base path and port, with defaults
    # And give a warning if the variable is not set
    if "REPORT_URL" not in os.environ:
        logger.warning("Environment variable REPORT_URL not set. Using default base path 'kooplex-query'. Set REPORT_URL to change this.")
    if "REPORT_PORT" not in os.environ:
        logger.warning("Environment variable REPORT_PORT not set. Using default port 9000. Set REPORT_PORT to change this.")
    base_path = _ge("REPORT_URL", "kooplex-query")
    base_path = base_path.strip("/")
    port = _ge("REPORT_PORT", "9000")

    args = [
        "streamlit",
        "run",
        str(script_path),
        "--server.port=" + port,
        "--server.address=0.0.0.0",
        "--server.baseUrlPath=" + base_path,
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
        "--browser.gatherUsageStats=false",
        "--server.headless=true",
        "--client.showErrorDetails=true",
    ]

    sys.argv = args
    sys.exit(stcli.main())


# Uncomment to debug session_state
#st.write(st.session_state)


if __name__ == '__main__':
    logger.info("Starting the application...")
    prompt()

    
