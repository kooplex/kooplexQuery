from typing import Annotated, TypedDict, Literal
import os

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, StructuredTool
from langchain_community.utilities.sql_database import SQLDatabase

from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
import pandas as pd


from dotenv import load_dotenv
load_dotenv("./sewage16-postgresql-connection-info.env")

# Define a search tool using DuckDuckGo API wrapper
search_DDG = StructuredTool.from_function(
        name="Search",
        func=DuckDuckGoSearchAPIWrapper().run,  # Executes DuckDuckGo search using the provided query
        description=f"""
        useful for when you need to answer questions about current events. You should ask targeted questions
        """,
    )

ollama_host = "http://wfct0p-ollamaapi:11434"

class MyToolSchema(BaseModel):
    question: str = Field(..., description="The user's question")

@tool(args_schema=MyToolSchema, infer_schema=False)
def explain_database_tool(question: str) -> str:
    """
    Useful when you need to answer question or retrieve information about the Sewage database 
    using the given database schema and table, column descriptions
    """
    
    from langchain.prompts import PromptTemplate
    
    # ollm = OllamaLLM(base_url=ollama_host, model="qwen2.5-coder:14b")
    ollm = OllamaLLM(base_url=ollama_host, model="llama3.1")
    prompt = PromptTemplate(
        template="""You are a helpful microbiologist who has access to a database containing DNA sequences from sewage samples
        Based on the following context: {context}
        Your task is to 
        - answer simply to the user's question
        - explain the meaning of the words in the context of the question
        - suggest alternative words that could be used in the question
    * Be concise

    Question: {question} 

    """,
        input_variables=["question", "context"],
    )

    dirname = "/v/projects/text2sql/david/Streamlit_chatgpt-app/"
    files = ["sewage_schema.sql", "table_column_description.txt", "sewage_data_descriptor.txt", "table_description.txt"]
    data = []
    for f in files:
        with open(dirname+f, 'r') as rr:
            data.append(rr.read())
    
    chain = prompt | ollm
    response = chain.invoke({"question":question, "context": data})
    # response = "The database is about samples collected from Mars"
    
    return response

@tool(args_schema=MyToolSchema, infer_schema=False)
def texttosql(question: str) -> str:
    """
    Useful when you need to from retrieve data from the Sewage database 
    using SQL queries and the user's question needs to be converted to SQL, such as listing rows, counting rows, or filtering data.
    """
    
    from texttosql_localllm.text2sql import Txt2Sql
    from texttosql_localllm.utils import postprocess_sql
    
    ddir = "/v/projects/text2sql/david/Streamlit_chatgpt-app"
    schema_file = f"{ddir}/sewage_schema.sql"
    examples_path = f"{ddir}/example_qa_sewage.csv"

    # Sewage database
    host=os.getenv("PG_HOST", 'localhost')
    port=os.getenv("PG_PORT", 5432)
    user=os.getenv("PG_USER", 'reader') 
    password=os.getenv("PG_PASSWORD", '')
    database = os.getenv("PG_DATABASE", 'sewage')
    schema = os.getenv("PG_SCHEMA", 'distilled')
    db = SQLDatabase.from_uri(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}", schema=schema)
    
    sql_llm = OllamaLLM(
        base_url=ollama_host,
        model="qwen2.5-coder:14b", 
        temperature=0.0, 
        keep_alive=1
    )
    
    t2s = Txt2Sql(sql_llm_model=sql_llm, db=db, schema_file=schema_file)
    t2s._init_embedding_model()
        
    t2s._init_examples()
    
    if len(t2s.vectorstore_examples.get()['ids']) == 0:
        examples_dict = pd.read_csv(examples_path)#.to_dict(orient='records')
        t2s.add_examples(examples_dict)
    t2s._init_example_selector()
    
    t2s._init_docs()
    if len(t2s.vectorstore_docs.get()['ids']) == 0:
        table_description = f"{ddir}/table_description.txt"
        table_columns_description = f"{ddir}/table_column_description.txt"
        t2s.load_split_add_csv(table_description, csv_args={'fieldnames': ['Table name', 'Description'], 'delimiter': '\t'})
        t2s.load_split_add_csv(table_columns_description, csv_args={'fieldnames': ['Table name', 'Column name', 'Variable type', 'Description'], 'delimiter': '\t'})
    
    # Add the database schema to the vector store
    t2s._init_dbschema()
    if len(t2s.vectorstore_dbschema.get()['ids']) == 0:
        t2s.add_dbschema()        

    t2s.set_question(question) 
    response, prompt = t2s.run_with_fewshot_prompt(return_prompt=False, full_dbschema=True)
    sql_gen = postprocess_sql(response['result'])
    return sql_gen


# List of tools that will be accessible to the graph via the ToolNode
tools = [explain_database_tool, texttosql]#, search_DDG]
tool_node = ToolNode(tools)

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed


# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
            max_retries=6,
    ).bind_tools(tools)
    # llm = ChatOpenAI(
    #     temperature=0.7,
    #     model="llama3.1",
    #     base_url="http://wfct0p-ollamaapi:11434/v1",
    #     openai_api_key="fdsfs",
    #     streaming=True,
    # **{'tool_choice':"any"}
    # ).bind_tools(tools)
    
  
    response = llm.invoke(messages)
    print(response.tool_calls)
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)

# Add conditional logic to determine the next step based on the state (to continue or to end)
graph.add_conditional_edges(
    "modelNode",
    should_continue,  # This function will decide the flow of execution
)
graph.add_edge("tools", "modelNode")

# Compile the state graph into a runnable object
graph_runnable = graph.compile()

# Function to invoke the compiled graph externally
def invoke_our_graph(st_messages, callables):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})
