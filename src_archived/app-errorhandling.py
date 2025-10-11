## Takes a user question and generates a SQL query using LLM
# and executes it on a database.
# Handles errors of the generation and the execution of the query
# TODO ??

import streamlit as st
from streamlit_extras.chart_container import *
from streamlit_elements import elements, mui, html

import sqlite3

import logging
logger = logging.getLogger(__name__)

import chromadb    
from db_sewage import DBQuery
from db_chat import DBChat     

from app_utils import *

init_step = 0


# ------- SET ENV VARIABLES --------------
if st.session_state.get("env_loaded") is None:
    init_step += 1
    logger.info(f"{init_step}. Loading environment variables")
    import os
    from dotenv import load_dotenv
    load_dotenv("/v/wfct0p/API-tokens/sewage16-postgresql-connection-info.env")
    st.session_state.env_loaded = True

openai_token = ""
with open("/v/wfct0p/API-tokens/openai-api.token") as f:
    openai_token = f.read().strip()
os.environ['OPENAI_API_KEY'] = openai_token



# Setup chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
    ]

if "sql_query" not in st.session_state:
    st.session_state.sql_query = None

if "user_input" not in st.session_state:
    st.session_state.user_input = None

if "error" not in st.session_state:
    st.session_state.error = None

if "config_initialized" not in st.session_state:
    init_step += 1
    logger.info(f"{init_step}. App is initializing Configs")
    chromadb.api.client.SharedSystemClient.clear_system_cache()   
    config = Config()
    st.session_state.config = config
    st.session_state.config_initialized = True
    logger.info("Config initialized")
## End of Session state initialization  --------------------

# def sync_examples():
#     sync_db_examplees_

st.set_page_config(layout="wide")

with st.sidebar:

    selected_sql_model_val = st.sidebar.selectbox(
    "Select model for Text2SQL",
    [e for e,m in enumerate(st.session_state.config.supported_models)],
    format_func=lambda x: st.session_state.config.supported_models[x].name,
)   
    selected_sql_model = st.session_state.config.supported_models[selected_sql_model_val]
    # selected_sql_model = st.selectbox("Select model for Text2SQL", ["qwen2.5-coder:14b", "vllm", "llama-3-groq-8B", "qwen:0.5b"])
    if selected_sql_model != st.session_state.config.sql_model:

        st.session_state.config.set_sql_llm(selected_sql_model)
        # st.session_state.config.set_sql_llm(ollama_sql)
        st.session_state.config.init_t2s()
        logger.info(f"Selected SQL model: {selected_sql_model}")
        st.sidebar.success(f"Selected SQL model: {selected_sql_model}")


# Display chat history
for msg in st.session_state['chat_history']:
        st.chat_message(msg['role']).write(msg['content'])
        
_examples = [
    'List all locations and the number of samples collected from that place!',
    'Count samples that have been collected above 13 Celsius!',
    "How many contigs contain the 'miaB' gene?",
]
user_prompt = st.chat_input("Enter a prompt") #, key="user_input")
if user_prompt:
    st.session_state['last_question']=user_prompt
else:
    # Text input for user question #hardcoded length!
    if st.session_state['chat_history'] == []:
        cols = st.columns(3)
        def _s(q):
            st.session_state['last_question']=q
        for i, q in enumerate(_examples):
            cols[i].button(q, type="secondary", on_click=_s, args=[q])

if 'last_question' in st.session_state:
    _q=st.session_state.pop('last_question')
    st.session_state['chat_history'].append({"role": "user", "content": _q})
    st.chat_message("user").write(_q)

    if st.session_state.chat_history[-1]['role'] == 'user':
        with st.spinner("Thinking..."): #with st.chat_message("assistant"):
            # Translate user question to SQL query
            # if st.session_state['chat_history']:
            try:
                sql_query = translate_to_sql(st.session_state['chat_history'], st.session_state.config.t2s)
                st.session_state.sql_query = sql_query  
                logger.info(f"SQL query generated:" )# {sql_query}")
            except:
                sql_query = "Error: Unable to generate SQL query. Please check your input."
                st.session_state['chat_history'].append({"role": "error", "content": sql_query})
                st.session_state.error = {"type": "generate", "content": sql_query}
                logger.error(f"Error generating SQL query: {sql_query}")
                st.sidebar.error(f"Error generating SQL query: {sql_query}")
                if st.sidebar.button("Correct Retry"):
                    st.session_state.config.t2s.init_llm(temperature=0.8)
                    st.session_state['last_question']=q
                    st.rerun()

                st.rerun()
            st.session_state['chat_history'].append({"role": "agent", "content": sql_query})
            # Save response to db

            #response = st.write_stream(())
            #response = st.markdown(sql_query)
            # st.rerun()

st.sidebar.radio("Autocorrect", options=["yes", "no"])
st.sidebar.radio("Always explain", options=["yes", "no"])

st.write(st.session_state.sql_query)
if st.session_state.sql_query:
    if st.sidebar.button("Execute SQL"):

        result = execute_sql_query(st.session_state.sql_query, st.session_state.config)
        if "error" in result.keys():
            st.session_state.error = {"type": "query", "content": result["error"]}
            logger.error(f"Error executing query: {result['error']}\n SQL: {st.session_state.sql_query}")
            #st.session_state['chat_history'].append({"role": "error", "content": result["error"]})
            st.sidebar.error(f"Error executing query: {result['error']}")
            
            # st.rerun()
        else:
            st.session_state.result_df = result["result"]
            st.session_state.error = None
                    
            if isinstance(st.session_state.result_df, str):
                st.write(st.session_state.result_df)
            else:
                # Display result dataframe
                st.dataframe(st.session_state.result_df )

    if st.session_state.error:
        if st.sidebar.button("Correct SQL query"):
            error_type, error_description = str(st.session_state.error['content']).split("(psycopg2.errors.")[1].split("LINE")[0].split(") ")
            # Correct SQL query
            st.session_state['last_question'] = f"Correct the SQL query because: {error_description} ({error_type})"
            st.session_state.sql_query = None
            st.session_state.error = None
            st.rerun()


# st.write(st.session_state['chat_history'])            
# Undo last question/instruction and the response to it
if st.sidebar.button("Undo last item"):
    if st.session_state['chat_history']:
        st.session_state['chat_history'].pop()
        st.session_state['chat_history'].pop()
        st.write(st.session_state['chat_history'])       
        # st.stop()
        # st.session_state.sql_query = None
        # st.session_state.error = None
        # st.session_state.result_df = None
        # st.session_state['last_question'] = None
        st.rerun()

# Explain the SQL query in relation to the user question and instructions
if st.sidebar.button("Explain"):
    if st.session_state.sql_query:
        st.session_state['chat_history'].append({"role": "user",
        "content": "Explain the SQL query in relation to the user question and instructions"})
        explanation = generate_sql_explanation(st.session_state['chat_history'], st.session_state.config.t2s)
        st.session_state['chat_history'].append({"role": "agent", "content": explanation})
        st.write(explanation)
    else:
        st.sidebar.error("No SQL query to explain.")


if st.sidebar.button("Determine"):

        st.session_state['chat_history'].append({"role": "user",
        "content": "Determine whether an SQL query need to be generated or just a simple text response!"})                
        query=f"{convert_to_text(st.session_state.chat_history)}"
        response = st.session_state.config.t2s.sql_llm_model.invoke(query)
        st.session_state['chat_history'].append({"role": "agent", "content": response})
        st.write(response)
