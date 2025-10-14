import time
from dataclasses import dataclass
import os
from typing import Iterator, List, Dict, Any
import logging
import re
from db_chat import DBChat
from db import DBQuery

logger = logging.getLogger(__name__)
_ge = lambda x,d: os.getenv(x, d)

@dataclass
class LLM_Model:
    type: str
    name: str

@dataclass
class Content_Chunk:
    type: str
    content: str

supported_models=[
    LLM_Model('ollama', 'llama3.1'),
    LLM_Model('ollama', 'qwen2.5-coder:3b'),
    LLM_Model('ollama', 'qwen2.5-coder:14b'),
    LLM_Model('vllm', 'Qwen/Qwen2.5-coder-3b'),
    LLM_Model('ollama', 'qwen2.5-coder:7b'),
#    LLM_Model('openai', 'gpt-3.5-turbo', None),
#    LLM_Model('openai', 'gpt-4.1-mini', None),
#    LLM_Model('openai', 'gpt-4.1', None),
    LLM_Model('ollama', 'hf.co/defog/sqlcoder-6b-2:Q5_K_M'),
]





#FIXME: PEP8, type safety, clarity

class Motor(object):
    def __init__(self, table_name_filter=None):
        from dotenv import load_dotenv
        load_dotenv("./config.env")

        self._table_name_filter=table_name_filter
        self.db_chat = self._dbchat_init()
        self.db_source = self._dbtarget_init()

    @property
    def error(self):
        return getattr(self, '_error', None)

    @property
    def current_model(self):
        return getattr(self, '_model', None)

    @current_model.setter
    def current_model(self, model_name):
        if self.current_model!=model_name:
            #from langchain_community.chat_models import ChatOpenAI
            from langchain_openai import ChatOpenAI
            logger.info (f"Changed to model {model_name}")
            _h=_ge("OLLAMA_HOST", "localhost")
            _p=int(_ge("OLLAMA_PORT", 11434))
            self._model=model_name
            self._llm_agent = ChatOpenAI(
                    temperature=0, openai_api_key="fdsfs", streaming=True, 
                    model_name=model_name, openai_api_base=f"http://{_h}:{_p}/v1")

    @property
    def sql(self):
        return getattr(self, '_sql', None)

    @sql.setter
    def sql(self, sql):
        self._sql=sql

    @property
    def df(self):
        return getattr(self, '_df', None)

    @df.setter
    def df(self, df):
        self._df=df

    @property
    def data_available(self):
        return self._df is not None

    @property
    def chat_history(self):
        return self._chat_history

    @property
    def question(self):
        return getattr(self, '_question', None)

    @property
    def can_prepare_save(self):
        return self.sql

    @property
    def can_save(self):
        return self.question and self.sql


    # public methods
    def new_session(self, username, email, label=None, referenced_session=None):
        import chromadb
        import random
        import pandas as pd
        from history import CustomChatHistory
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        label=label or random.randbytes(16)
        data_descriptor = self.db_chat.load_knowledge(reference='data_descriptor')
        dbschema = self.db_chat.load_knowledge(reference='schema')
        dbreference = self.db_chat.load_knowledge(reference='reference')
        table_description=dict(self.describe_tables())
        table_column_description = pd.DataFrame(self.describe_columns())
        context = f"""{data_descriptor}
            
{dbreference} table descriptions: {table_description}

{dbreference} table column descriptions: {table_column_description}

Database Schema: {dbschema}
        """
        self._chat_history=CustomChatHistory(context)
        self._chat_history.add_system_message(self.db_chat.load_knowledge(reference='instruction'))
        self._plot_history=CustomChatHistory()
        self._plot_history.add_system_message("You are a helper to create plots")
        chromadb.api.client.SharedSystemClient.clear_system_cache() #TODO test if really required parrallel runs
        self.session_id=self.db_chat.new_session(username=username, email=email, label=label, meta="", referenced_session=referenced_session)
        logger.info ("NEW SESSION")
        return self.session_id

    def describe_tables(self):
        return self.db_source.describe_tables(self._table_name_filter)

    def describe_columns(self):
        return self.db_source.describe_columns(self._table_name_filter)

    def pop(self):
        self._chat_history.pop()
        self._sql=None

    def fetch_examples(self, n=3):
        return self.db_chat.fetch_examples(n)

    def select_example(self, question, sql):
        if self._chat_history.is_empty:
            self._chat_history.add_user_message(question, metadata={'type': 'example_question'})
            self._chat_history.add_ai_message(sql, metadata={'type': 'example_sql', 'parsed': [Content_Chunk('sql', sql)]})
            self.sql=sql


    async def chat(self, prompt, model_name='llama3.1'):
        t0=time.time()
        self.current_model=model_name
        self._chat_history.add_user_message(prompt, metadata={'timestamp': t0, 'model': self.current_model })
        collected=""
        async for chunk in self._llm_agent.astream(self._chat_history.messages):
            if c:=chunk.content:
                collected += c
                yield c
        self._chat_history.add_ai_message(collected, metadata={'type': 'generated', 'duration': time.time()-t0, 'parsed': self._parse_sql(collected) })
        self.db_chat.save_chat_item(self.session_id, prompt, collected, self.current_model)

    async def correct_error(self, error, model_name='llama3.1'):
        detail=getattr(error, 'orig', None)
        statement=getattr(error, 'statement', None)
        prompt = f"Correct the SQL query\n{statement}\nbecause: {detail}"
        t0=time.time()
        self.current_model=model_name
        self._chat_history.add_user_message(prompt, metadata={'timestamp': t0, 'model': self.current_model })
        collected=""
        async for chunk in self._llm_agent.astream(prompt):
            if c:=chunk.content:
                collected += c
                yield c
        self._chat_history.add_ai_message(collected, metadata={'type': 'generated', 'duration': time.time()-t0, 'parsed': self._parse_sql(collected) })

    async def plot(self, instruction_prompt, model_name='qwen2.5-coder:3b'):
        if self.df is not None:
            t0=time.time()
            self.current_model=model_name
            self._chat_history.add_user_message(instruction_prompt, metadata={'type': 'plot', 'model': self.current_model})

            # PLOT WITH LLM  FIXME
            if self.df.shape[0] < 1000:
                tmpdf = self.df.copy()
            else:
                tmpdf = self.df.sample(1000, random_state=42)
            prompt = f"""
Data: {tmpdf.to_dict()}

User instructions for plotting: {instruction_prompt}

* Refer to the data as 'df'
* If there are multiple plots then use subplots in matplotlib
            """
            self._plot_history.add_user_message(prompt)
            resp = ""
            async for chunk in self._llm_agent.astream(self._plot_history.messages):
                if c:=chunk.content:
                    resp += c
                    yield c
            self._plot_history.add_ai_message(resp)
            # Extract the python code from the response
            code = resp.split("```python")[1].split("```")[0].strip()
            # Execute the code
            local_scope = {}
            noshow_code = "\n".join(
                f"# {line}" if "show" in line or "df =" in line else line
                for line in code.splitlines()
            )
            duration=time.time()-t0
            try:
                exec(noshow_code , {'df': self.df}, local_scope)  # Execute the code in a local scope
                fig=local_scope.get('fig', None)  # Get the figure from the local scope
                fig_type='plotly_chart'
                if fig is None:
                    fig=local_scope.get('plt', None)
                    fig_type='pyplot'
                self._chat_history.add_ai_message('figure generated', metadata={'content': fig, 'type': fig_type, 'code': code, 'duration': duration})
            except Exception as e:
                logger.error(e)
                self._chat_history.add_ai_message(str(e), metadata={'content': e, 'type': 'error', 'code': code, 'duration': duration})

    async def prepare_save(self, model_name='llama3.1'):
        # Function to generate a the real question based on the chat history the SQL response
        h=[]
        for r in self.chat_history.filter(['plot']):
            h.append(r['question'])
            h.append(r['answer'])
        h='\n'.join(h)
        prompt=f"""For last SQL {self.sql} based on the chat history\n{h}\nwhat was the question this sql query belongs to? 
        If the final question is not in the chat history, please generate a new question else repeat the question from the chat history.
        Pay attention to user changes to SQL code and match it to the question.
        Just provide the question without any extra explanation.
        """
#logger.info(f"Generating THE question with: {prompt}")
        self.current_model=model_name
        collected=""
        async for chunk in self._llm_agent.astream(prompt):
            if c:=chunk.content:
                collected += c
                yield c
        logger.info(f"Generated question: {collected}")
        self._question=collected#.split('```', 2)[1].strip()


    def execute(self, sql):
        try:
            self._error=None
            self._chat_history.add_user_message(sql, metadata={'type': 'submit_sql'})
            self.sql=sql
            t0=time.time()
            df=self.db_source.query_to_df(sql)
            self.df=df
            self._chat_history.add_ai_message(df.head().to_string(), metadata={'type': 'dataframe', 'dataframe': df, 'query': sql, 'duration': time.time()-t0})
        except Exception as e:
            self._chat_history.add_ai_message(str(e), metadata={'type': 'error', 'content': e, 'query': sql})
            self._error=e
            self.sql=None

    def save_query(self):
        print("Saving")
        self.db_chat.save_query(self.session_id, self.question, self.sql, question_type = 'user', public = False)
        self._question=None

    # private methods
    def _dbchat_init(self) -> DBChat:
        return DBChat(
            hostname=_ge('CHAT_HOST', 'localhost'), port=int(_ge('CHAT_PORT', 5432)),
            database=_ge('CHAT_DATABASE', 'sewage'), schema=_ge('CHAT_SCHEMA', 'chat'),
            db_user=_ge('CHAT_USER', 'chat_agent'),
            db_password=_ge('CHAT_PASSWORD', ''), generated_callback=lambda c: None
        )

    def _dbtarget_init(self) -> DBQuery:
        return DBQuery(
            hostname=_ge('PG_HOST', 'localhost'), port=int(_ge('PG_PORT', 5432)),
            database=_ge('PG_DATABASE', 'sewage'), schema=_ge('PG_SCHEMA', 'distilled'),
            db_user=_ge('PG_USER', 'reader'),
            db_password=_ge('PG_PASSWORD', '')
        )

    def _parse_sql(self, content: str) -> [Content_Chunk]:
        rest=content
        chunks=[]
        while True:
            try:
                before, rest = rest.split("```sql", 1)
                chunks.append(Content_Chunk("txt", before))
                q, rest=rest.split("```", 1)
                chunks.append(Content_Chunk("sql", q))
                if self.sql is None:
                    self.sql=q
            except ValueError:
                if rest:
                    chunks.append(Content_Chunk("txt", rest))
                break
        return chunks




if __name__ == '__main__':
    import asyncio
    import sys
    def print_stream(g):
        async def run():
            async for chunk in g:
                print (chunk, end='', flush=True)
        asyncio.run(run())     

    m=Motor()

    # initialize session
    sid=m.new_session(username='mock', email='mock@test.bla')
    print(sid)

    # test code fix
    sql="""
SELE CT 
  l.plant AS plant_name,
  COUNT(m.id) AS num_samples
FROM 
  distilled.meta m
JOIN 
  distilled.location l ON m.location_id = l.id
GROUP BY 
  l.plant;
    """
    m.execute(sql)
    if error:=m.chat_history.messages_with_meta[-1].metadata.get('error'):
        print ("OOPS", error)
        print_stream(m.correct_error(error))
        print(m.chat_history.messages_with_meta[-1].metadata.get('parsed'))


    # test chat
    print_stream(m.chat("count samples per plant"))

    # parse response
    if parsed:=m.chat_history.messages_with_meta[-1].metadata.get('parsed'):
        for p in parsed:
            if p.type=='sql':
                sql=p.content
                print ("Runnning", sql)
                m.execute(sql)
                break
    else:
        print ("No sql found in response")
        sys.exit()

    # try plotting
    print_stream(m.plot("plot as piechart"))

    if fig:=m.chat_history.messages_with_meta[-1].metadata.get('fig'):
        #fig.show()
        print(fig)
