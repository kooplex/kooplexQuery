from langchain_ollama import OllamaLLM
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from openai import OpenAI
from dataclasses import dataclass

import time, os, re, ast
import sql_metadata
import pandas as pd

import sqlite3

from typing import List, Tuple

from texttosql_localllm.text2sql import Txt2Sql
from texttosql_localllm.utils import postprocess_sql

import logging
logger = logging.getLogger(__name__)

# SQLite database setup
# We save the database configurations in a SQLite database
def init_sqlite_db():
    conn = sqlite3.connect('database_configurations.db')
    cursor = conn.cursor()
    # Create table for database configurations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS db_configurations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            host TEXT NOT NULL,
            port INTEGER NOT NULL,
            user TEXT NOT NULL,
            password TEXT NOT NULL,
            database_name TEXT NOT NULL,
            schema TEXT NOT NULL
        )
    ''')
    # Create table for related documents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS related_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_content BLOB NOT NULL,
            FOREIGN KEY (config_id) REFERENCES db_configurations (id)
        )
    ''')
    conn.commit()
    conn.close()



@dataclass
class LLM_Model:
    type: str
    name: str
    host: str


class DatabaseConfig:
    def __init__(self, name, type, host, port, user, password, database_name, schema):
        self.name = name
        self.type = type
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database_name = database_name
        self.schema = schema

    def __repr__(self):
        return f"DatabaseConfig({self.name}, {self.type}, {self.host}, {self.port}, {self.user}, {self.password}, {self.database_name}, {self.schema})"

    # Function to load database configurations from SQLite
    def from_db(self, db_name):
        conn = sqlite3.connect('configurations.db')
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM db_configurations WHERE name = {db_name}")
        row = cursor.fetchone()
        if row:
            return DatabaseConfig(*row[1:])
        else:
            raise ValueError(f"Database configuration '{db_name}' not found.")
        
# For saving database configs -------------------------------


class Config:
    """Class to setup the database connection for the dataset and configure LLM models/the texttosql module for the application."""

    def __init__(self):
        self.table_names = []
        self.column_names = []

        # Sewage database
        host=os.getenv("PG_HOST", 'localhost')
        port=os.getenv("PG_PORT", 5432)
        user=os.getenv("PG_USER", 'reader') 
        password=os.getenv("PG_PASSWORD", '')
        database = os.getenv("PG_DATABASE", 'sewage')
        schema = os.getenv("PG_SCHEMA", 'distilled')

        self.agent_model = None
        self.sql_model = None

        # OLLAMA host
#FIXME hardcoded 
        self.ollama_host = "http://wfct0p-ollamaapi:11434"
        self.vllm_host = "http://wfct0p-vllm:8000/v1"

        #!!!!! Initialize the database connection saved in sqlite!!!!!!!        
        self.db = SQLDatabase.from_uri(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}", schema=schema)
        #self.db = SQLDatabase.from_uri(f"sqlite:///../Chinook/Chinook.db")  # Replace with your actual database path
        
        # The path to the directory where the data files are stored for initialization
#FIXME
        self.ddir = "."
        self.schema_file = f"{self.ddir}/sewage_schema.sql"
        self.examples_path = f"{self.ddir}/example_qa_sewage.csv"
        self.t2s = None 
        # self.init_t2s()
        print("Config initialized")

    def set_llm_agent(self, agent_model):
        # Set the LLM agent with the specified model and host
        self.agent_model = agent_model
        self.llm_agent = ChatOpenAI(temperature=0, streaming=True, model_name=self.agent_model, openai_api_base=self.ollama_host+"/v1")  

    def set_sql_llm(self, sql_model=None):
        # Set the SQL LLM with the specified model and host
        # Check if the model is "vllm" or "ollama"
        logger.info(f"Setting SQL LLM model: {sql_model.name}, {sql_model.type}, {sql_model.host}")
        self.sql_model = sql_model.name
        if sql_model.type == "vllm":
            ollama_sql = VLLMOpenAI(
                openai_api_key="EMPTY",
                openai_api_base=sql_model.host,
                max_tokens=88,
                model_name=self.sql_model,
                temperature=0.0,
                tensor_parallel_size=3,
            )
            self.sql_llm = ollama_sql
        elif sql_model.type == "openai":
            self.sql_llm = ChatOpenAI(temperature=0, 
                                      model_name=self.sql_model)  

        elif sql_model.type == "ollama":
            # Set the SQL LLM with the specified model and host 
            logger.info(f"OLL Setting SQL LLM model: {sql_model.name}, {sql_model.type}, {sql_model.host}")
            self.sql_llm = OllamaLLM(base_url=sql_model.host,
                                     model=self.sql_model, 
                                     temperature=0.0, 
                                     keep_alive=1)
            


    def init_t2s(self):
        # Initialize the Txt2Sql object with the database and schema file
        # and the examples 
        # The databse schema is added from file
        self.t2s = Txt2Sql(sql_llm_model=self.sql_llm, db=self.db, schema_file=self.schema_file)
        self.t2s._init_embedding_model()
        
        self.t2s._init_examples()
        
        if len(self.t2s.vectorstore_examples.get()['ids']) == 0:
            examples_dict = pd.read_csv(self.examples_path)#.to_dict(orient='records')
            self.t2s.add_examples(examples_dict)
        self.t2s._init_example_selector()
        
        self.t2s._init_docs()
        if len(self.t2s.vectorstore_docs.get()['ids']) == 0:
            table_description = f"{self.ddir}/table_description.txt"
            table_columns_description = f"{self.ddir}/table_column_description.txt"
            self.t2s.load_split_add_csv(table_description, csv_args={'fieldnames': ['Table name', 'Description'], 'delimiter': '\t'})
            self.t2s.load_split_add_csv(table_columns_description, csv_args={'fieldnames': ['Table name', 'Column name', 'Variable type', 'Description'], 'delimiter': '\t'})
        
        # Add the database schema to the vector store
        self.t2s._init_dbschema()
        if len(self.t2s.vectorstore_dbschema.get()['ids']) == 0:
            self.t2s.add_dbschema()
            
    
    def get_table_and_column_names(self):
        # Retrieve all table names
        self.table_names = self.db.get_table_names()
        columns = []
        # Iterate through tables and list column names
        for table in self.table_names:
            query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}';"
            s=self.db.run(query)
            columns.extend(re.sub(r"[^a-zA-Z\w\s]", "", s).split())

        self.column_names = set(columns)

    @property
    def supported_models(self):
        # Return a list of supported models
        return [LLM_Model('ollama', 'qwen2.5-coder:14b', self.ollama_host),
                LLM_Model('ollama', 'qwen2.5-coder:32b', self.ollama_host),
                LLM_Model('ollama', 'qwen2.5-coder:7b', self.ollama_host),
                LLM_Model('ollama', 'qwen2.5-coder:3b', self.ollama_host),
                LLM_Model('ollama', 'qwen2.5-coder:14b-instruct-q4_K_M', self.ollama_host),
                LLM_Model('openai', 'gpt-3.5-turbo', None),
                LLM_Model('openai', 'gpt-4.1-mini', None),
                LLM_Model('openai', 'gpt-4.1', None),
                LLM_Model('vllm', 'TechxGenus/deepseek-coder-6.7b-instruct-AQLM', self.vllm_host),
                LLM_Model('vllm', 'TechxGenus/deepseek-coder-33b-base-AQLM', self.vllm_host),
                #LLM_Model('vllm', 'jun2114/Llama31-8B-oci-text2sql-no-think', self.vllm_host),                
                LLM_Model('ollama', 'hf.co/defog/sqlcoder-7b-2:Q5_K_M', self.ollama_host)
        ]
            
        
    
def convert_to_text(history):
    # Function to convert the chat history json to simple text
    converted_text = ""
    for item in history:
        # converted_text += f"\nUser: {item['question']}\n"
        # converted_text += f"sql query: {item['sql_query']}\n"
        converted_text += item['content']
    return converted_text   

# Function to translate user question to SQL query (mock/dummy LLM tool)
def translate_to_sql(history, t2s, additional_info=None, prompt_only=False):
    # t2s.set_question(question) 
    logger.info(f"t2s tool Question: {history:}")
    t2s.set_question(convert_to_text(history)) 
    t2s.add_additional_info(additional_info)
    
    response, prompt = t2s.run_with_fewshot_prompt(return_prompt=prompt_only, full_dbschema=True)
    if prompt_only:
        return prompt
    else:
        sql_gen = postprocess_sql(response['result'])
        logger.info(f"t2s tool SQL generated:") # {prompt}")
        # sql_gen = "SELECT artista, idx FROM album"
    
        return sql_gen
    
def generate_response_openai(prompt, model):
    openai_client = OpenAI()

    messages=[
        {"role": "developer", "content": "You are an SQL programming expert."},
        {
            "role": "user",
            "content": prompt,
        },]
    completion = openai_client.chat.completions.create(model=model, messages=messages)
    # completion = openai_client.chat.completions.create(model=model, messages=messages)
    response = completion.choices[0].message.content

    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens

# Function to execute SQL query on the database
def execute_sql_query(query, db_config):
    import datetime
    try:
        res = db_config.db.run(query)
        # Convert the string to a list of tuples
        try:
            data_dict = eval(res)
            # data_dict = ast.literal_eval(res)

            # Create a DataFrame from the list of tuples
            result = pd.DataFrame(data_dict)
        except:
            result = res
    except Exception as e:
        return {"success": False, "error": str(e)}
    return {"success":True, "result": result}

# Function to plot a figure from the result
def plot_result(df, streamlit):
    # import sketch
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    
    # df.plot(kind='bar', ax=ax)
    # streamlit.pyplot(fig)
    streamlit.markdown("## Further advise on how to plot the data:")
    advise = df.sketch.ask("what is the best way to plot this?", call_display=False)
    streamlit.write(advise)
    howto = df.sketch.howto("Plot this data in a suitable manner", call_display=False)
    streamlit.write(howto)
    # streamlit.pyplot(eval(howto))

def validate_sql(query: str, db_config: SQLDatabase, existing: bool = False) -> Tuple[List[str], List[str]]:
    """ existing: if True then we check if the tables and columns are in the database, if False we check if they are not in the database """
    logger.info(f"Validating SQL query: {query}")
    p = sql_metadata.Parser(query)
    highlight_wrong_tables = []
    for table in p.tables:
        if table not in db_config.get_usable_table_names() * (not existing):
            highlight_wrong_tables.append(table)
    highlight_wrong_columns = []
    for column in p.columns:
        if column not in ["id", "AlbumId", "Title", "ArtistId"] * (not existing):
            highlight_wrong_columns.append(column)
    return highlight_wrong_tables, highlight_wrong_columns
    

def add_highlights(response_sentences, hwt, hwc, bg="red", text_t="blue", text_c="orange"):
    highlighted_text = []
    for word in response_sentences.split():
        if word.split(",")[0] in hwt:
            highlighted_text.append(f":{text_t}[:{bg}-background[" + word + "]]")
        elif word.split(",")[0] in hwc:
            highlighted_text.append(f":{text_c}[:{bg}-background[" + word + "]]")
        else:
            highlighted_text.append(word)
    return highlighted_text

def generate_sql_explanation(history, t2s):
    # Function to generate a follow-up question based on the SQL response
    query=f"{convert_to_text(history)}"
    response = t2s.sql_llm_model.invoke(query)
    return response
    

def generate_followup_question(history, sql_response, llm_agent):
    # Function to generate a follow-up question based on the SQL response
    # Use the LLM agent to generate a follow-up question
    query=f"Based on the chat history {convert_to_text(history)}, what would be a good follow-up question?"
    logger.info(f"Generating follow-up question with query: {query}")
    followup_question = llm_agent.invoke(query)
    return followup_question

def generate_the_question(history, sql_response, llm_agent):
    # Function to generate a the real question based on the chat history the SQL response
    query=f"""Based on the chat history {convert_to_text(history)}, what was the question to which this sql query belongs to? 
    If the final question is not in the chat history, please generate a new question else repeat the question from the chat history.
    The format sof the response is: '```THE_QUESTION <THE_QUESTION>```
    Exaplanation and details of the question belonging to the final sql query: <EXPLANATION>"""
    logger.info(f"Generating THE question with query: {query}")
    # return llm_agent.invoke(query).content
    response = llm_agent.invoke(query).content
    logger.info(f"Generated question: {response}")
    generated_question = response.split("```THE_QUESTION")[1].split("```")[0]
    # generated_question = response.split("```")[1].split("\n")[0]
    explanation = response.split("```THE_QUESTION")[1].split("```")[1]
    # explanation = response.split("```")[1]
    return generated_question, explanation


def translate_to_sql_fake(chat_history):
    # Use FakeLLM to generate a SQL query from the latest user message
    user_message = chat_history[-1]['content']
    # For demo, just echo a fake SQL query
    return fakellm(user_message)

def generate_sql_explanation_fake(chat_history):
    # Use FakeLLM to generate an explanation for the SQL query
    sql_query = chat_history[-1]['content']
    return fakellm(sql_query)


## PATCH --------------------------------------
## For callback's we need to patch streamlit, because streamlit-elements fails to be compatible above version 1.34 an 1.40
def patch_modules_streamlit_elements(file: str, old_line: str, new_line: str):
    import streamlit_elements
    import os
    


    relative_file_path = "core/callback.py"
    library_root = list(streamlit_elements.__path__)[0]
    file_path = os.path.join(library_root, relative_file_path)

    with open(file_path, "r") as file:
        lines = file.readlines()

    is_changed = False
    for index, line in enumerate(lines):
        if old_line in line:
            print(f"Replacing line {index + 1} in {file_path}")
            lines[index] = line.replace(old_line, new_line)
            is_changed = True

    if is_changed:
        with open(file_path, "w") as file:
            file.writelines(lines)
        import importlib
        importlib.reload(streamlit_elements)

    return True

def patch_streamlit_elements():
    # fix 1.34.0
    patch_modules_streamlit_elements(
        "core/callback.py",
        "from streamlit.components.v1 import components",
        "from streamlit.components.v1 import custom_component as components\n",
    )


    #fix 1.40.0
    patch_modules_streamlit_elements(
        "core/callback.py",
        '        user_key = kwargs.get("user_key", None)\n',
        """
        try:
            user_key = None
            new_callback_data = kwargs[
                "ctx"
            ].session_state._state._new_session_state.get(
                "streamlit_elements.core.frame.elements_frame", None
            )
            if new_callback_data is not None:
                user_key = new_callback_data._key
        except:
            user_key = None
        """.rstrip()
        + "\n",
    )

## It needs to be called in the app
## if __name__ == "__main__":
##      patch_streamlit_elements()

## PATCH --------------------------------------
