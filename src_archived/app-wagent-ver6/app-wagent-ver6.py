## Takes a user question and generates a SQL query using LLM
# and executes it on a database.
# Handles errors of the generation and the execution of the query
# TODO ??

import streamlit as st
from streamlit_extras.chart_container import *
from streamlit_elements import elements, mui, html
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

import sqlite3

import logging
logger = logging.getLogger(__name__)

import chromadb    
from db_sewage import DBQuery
from db_chat import DBChat     

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

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

st.set_page_config(layout="wide")


# Setup chat history
# if 'chat_history' not in st.session_state:
#     st.session_state['chat_history'] = [
#     ]

if "chat_history" not in st.session_state:
    files = ["sewage_schema.sql", "table_column_description.txt", "sewage_data_descriptor.txt", "table_description.txt"]
    data = []
    for f in files:
        with open(f, 'r') as rr:
            data.append(rr.read()[:8])
    # st.write(data)
    st.session_state.data = data
    
    # st.session_state['chat_history'] = [{'role':'assistant', 'content': st.session_state.data }]
    st.session_state['chat_history'] = [AIMessage(content=st.session_state.data )]


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

# if "chat_history" not in st.session_state:
#     # default initial message to render in message state
#     # st.session_state["chat_history"] = [AIMessage(content="How can I help you?")]
#     st.session_state["chat_history"] = [{"role": "agent", "content": "How can I help you?"}]

# st.write(st.session_state['chat_history'])
# Display chat history
for msg in st.session_state['chat_history'][1:]:
        # st.write(msg)
        # st.chat_message(msg)
        # st.chat_message(msg['role']).write(msg['content'])
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)
        
_examples = [
    'How many MAGs do we have with contamination below 10% and GC content greater than 0.6?',
    'Count samples that have been collected above 13 Celsius!',
    "what is the database about?",
]
user_prompt = st.chat_input("Choose a question ro ask your own") #, key="user_input")
if user_prompt:
    st.session_state['last_question']=user_prompt
else:
    # Text input for user question #hardcoded length!
    if len(st.session_state['chat_history']) == 1:
        cols = st.columns(3)
        def _s(q):
            st.session_state['last_question']=q
        for i, q in enumerate(_examples):
            cols[i].button(q, type="secondary", on_click=_s, args=[q])

if 'last_question' in st.session_state:
    _q=st.session_state.pop('last_question')
    # st.session_state['chat_history'].append({"role": "user", "content": _q})
    st.session_state['chat_history'].append(HumanMessage(content=q ))
    st.chat_message("user").write(_q)

    if type(st.session_state.chat_history[-1]) == HumanMessage:
        with st.chat_message("assistant"):
            msg_placeholder = st.empty()  # Placeholder for visually updating AI's response after events end
            # create a new placeholder for streaming messages and other events, and give it context
            st_callback = get_streamlit_cb(st.empty())
            response = invoke_our_graph(st.session_state.chat_history, [st_callback])
            last_msg = response["messages"][-1].content
            # st.session_state.chat_history.append({"role": "ai", "content": last_msg})  # Add that last message to the st_message_state
            st.session_state['chat_history'].append(AIMessage(content= last_msg ))
            msg_placeholder.write(last_msg) # visually refresh the complete response after the callback container
            # sql_query = translate_to_sql(st.session_state['chat_history'], st.session_state.config.t2s)

            # st.session_state.sql_query = sql_query  
        


# Button to create a new session and clear history
if st.sidebar.button("New Session"):
    st.session_state['chat_history'] = []  # Clear current history
    st.session_state.stage = "user"  # Reset stage
    st.session_state.user_input = ""  # Clear any user input
    st.session_state.sql_query = ""  # Clear SQL query
    #st.session_state.sessions.append({"question": "", "sql_query": ""})  # Append new empty session to sessions
    st.rerun()  # Rerun the app to reset everything

st.sidebar.radio("Autocorrect", options=["no", "yes"])
st.sidebar.radio("Always explain", options=["no", "yes"])

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
        st.session_state['chat_history'].append(HumanMessage(content= "Explain the SQL query in relation to the user question and instructions" ))
    
        explanation = generate_sql_explanation(st.session_state['chat_history'], st.session_state.config.t2s)
        st.session_state['chat_history'].append({"role": "ai", "content": explanation})
        st.write(explanation)
    else:
        st.sidebar.error("No SQL query to explain.")


if st.sidebar.button("Determine"):

        st.session_state['chat_history'].append({"role": "user",
        "content": "Determine whether an SQL query need to be generated or just a simple text response!"})                
        query=f"{convert_to_text(st.session_state.chat_history)}"
        response = st.session_state.config.t2s.sql_llm_model.invoke(query)
        st.session_state['chat_history'].append({"role": "ai", "content": response})
        st.write(response)
