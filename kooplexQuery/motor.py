import time
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
from typing import Iterator, List, Dict, Any
import logging
import re
from urllib.parse import urlparse
from kooplexQuery.db_chat import DBChat
from kooplexQuery.db import DBQuery
from kooplexQuery.utils.sync_manager import VectorStoreSyncManager

logging.basicConfig(
    filename='/tmp/app.log', 
    level=logging.DEBUG
)
# Suppress verbose file-watcher debug noise from Streamlit/watchdog.
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
_ge = lambda x,d: os.getenv(x, d)

@dataclass
class LLM_Model:
    provider: str
    model_name: str

@dataclass
class Content_Chunk:
    type: str
    content: str


@dataclass
class _TextChunk:
    content: str


@dataclass
class _TextResponse:
    content: str

if not os.getenv('DEFAULT_LLM_MODEL') and not os.getenv('DEFAULT_LLM_PROVIDER'):
    logger.warning("No default LLM model or provider set in environment variables. Using fallback default: gpt-4.1-mini from openai. Set DEFAULT_LLM_MODEL and DEFAULT_LLM_PROVIDER to change this.")
DEFAULT_LLM = LLM_Model(
    provider=os.getenv('DEFAULT_LLM_PROVIDER', 'openai'),
    model_name=os.getenv('DEFAULT_LLM_MODEL', 'gpt-4.1-mini')
)
logger.info("Default LLM set to: %s provider: %s", DEFAULT_LLM.model_name, DEFAULT_LLM.provider)


def _llm_models_db_path() -> Path:
    return Path(os.getenv('LLM_MODELS_DB_PATH', os.path.join(os.getcwd(), 'llm_models.sqlite3')))


def _normalize_ollama_base_url(provider: str | None = None) -> str:
    raw_provider = provider or _ge('OLLAMA_HOST', 'http://localhost')
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', raw_provider):
        raw_provider = f'http://{raw_provider}'
    # parsed = urlparse(raw_provider)
    normalized = raw_provider.rstrip('/')
    if not normalized.endswith('/v1'):
        normalized = f'{normalized}/v1'
    return normalized


def is_provider_url(provider: str) -> bool:
    return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', provider))


def _looks_like_azure_openai_url(url: str) -> bool:
    candidate = (url or '').strip()
    if not candidate:
        return False

    configured = _ge(
        'AZURE_FOUNDRY_URL',
        _ge('AZURE_BASE_URL', _ge('AZURE_OPENAI_ENDPOINT', '')),
    ).strip()

    def _normalize(raw: str) -> str:
        value = (raw or '').strip().rstrip('/')
        if value.endswith('/openai'):
            value = value[:-len('/openai')]
        return value

    candidate_norm = _normalize(candidate)
    configured_norm = _normalize(configured)
    if configured_norm and (candidate_norm == configured_norm or candidate_norm.startswith(configured_norm) or configured_norm.startswith(candidate_norm)):
        return True

    try:
        parsed = urlparse(candidate_norm)
        host = (parsed.hostname or '').lower()
    except Exception:
        host = ''
    return host.endswith('.services.ai.azure.com') or host.endswith('.openai.azure.com')


def _has_azure_openai_config() -> bool:
    return bool(
        _ge('AZURE_FOUNDRY_API_KEY', _ge('AZURE_API_KEY', _ge('AZURE_OPENAI_API_KEY', ''))).strip()
        and _ge('AZURE_FOUNDRY_URL', _ge('AZURE_BASE_URL', _ge('AZURE_OPENAI_ENDPOINT', ''))).strip()
    )


def _has_azure_anthropic_config() -> bool:
    return bool(
        _ge('AZURE_ANTHROPIC_API_KEY', _ge('AZURE_API_KEY', _ge('ANTHROPIC_API_KEY', ''))).strip()
        and _ge('AZURE_ANTHROPIC_BASE_URL', '').strip()
    )


class _AzureAnthropicAdapter:
    def __init__(self, model_name: str):
        try:
            import importlib
            Anthropic = importlib.import_module('anthropic').Anthropic
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Provider 'azure-anthropic' requires the 'anthropic' package. "
                "Install it in your runtime environment, e.g. pip install anthropic."
            ) from exc

        self.model_name = model_name
        self.api_key = _ge('AZURE_ANTHROPIC_API_KEY', _ge('AZURE_API_KEY', _ge('ANTHROPIC_API_KEY', ''))).strip()
        self.base_url = _ge('AZURE_ANTHROPIC_BASE_URL', '').strip()
        self.max_tokens = int(_ge('AZURE_ANTHROPIC_MAX_TOKENS', '1024'))

        if not self.api_key or not self.base_url:
            raise ValueError(
                "Provider 'azure-anthropic' requires AZURE_ANTHROPIC_BASE_URL and "
                "AZURE_ANTHROPIC_API_KEY (or AZURE_API_KEY)."
            )

        self.client = Anthropic(base_url=self.base_url, api_key=self.api_key)

    def _to_messages(self, payload):
        if isinstance(payload, str):
            return '', [{'role': 'user', 'content': payload}]

        system_parts = []
        messages = []
        for msg in payload:
            mtype = getattr(msg, 'type', None)
            content = getattr(msg, 'content', '')
            if isinstance(content, list):
                content = ''.join(str(part) for part in content)
            content = str(content or '')
            if not content:
                continue
            if mtype == 'system':
                system_parts.append(content)
            elif mtype in {'ai', 'assistant'}:
                messages.append({'role': 'assistant', 'content': content})
            else:
                messages.append({'role': 'user', 'content': content})
        if not messages:
            messages = [{'role': 'user', 'content': 'Hello'}]
        return '\n\n'.join(system_parts), messages

    def _invoke_sync(self, payload) -> str:
        system, messages = self._to_messages(payload)
        kwargs = {
            'model': self.model_name,
            'max_tokens': self.max_tokens,
            'messages': messages,
        }
        if system:
            kwargs['system'] = system
        response = self.client.messages.create(**kwargs)
        parts = []
        for block in getattr(response, 'content', []) or []:
            text = getattr(block, 'text', None)
            if text:
                parts.append(text)
        return ''.join(parts)

    async def ainvoke(self, payload):
        text = await asyncio.to_thread(self._invoke_sync, payload)
        return _TextResponse(content=text)

    async def astream(self, payload):
        text = await asyncio.to_thread(self._invoke_sync, payload)
        yield _TextChunk(content=text)


def _normalize_model_provider(provider: str | None, model_name: str | None = None) -> str:
    normalized_provider = (provider or '').strip()
    if not normalized_provider:
        if model_name and model_name.startswith('gpt-'):
            return 'azure-openai' if _has_azure_openai_config() else 'openai'
        if model_name and model_name.startswith('claude-'):
            return 'azure-anthropic' if _has_azure_anthropic_config() else 'anthropic'
        return _normalize_ollama_base_url()

    lowered = normalized_provider.lower()
    if lowered in {'azure', 'azure-openai', 'azure-openai-foundry'}:
        return 'azure-openai'
    if lowered in {'azure-anthropic', 'anthropic-azure', 'azure-claude'}:
        return 'azure-anthropic'
    if is_provider_url(normalized_provider):
        if model_name and model_name.startswith('gpt-') and _has_azure_openai_config() and _looks_like_azure_openai_url(normalized_provider):
            return 'azure-openai'
        return _normalize_ollama_base_url(normalized_provider)
    return normalized_provider


def default_provider_for_model(model_name: str) -> str:
    if model_name.startswith('gpt-'):
        return 'azure-openai' if _has_azure_openai_config() else 'openai'
    if model_name.startswith('claude-'):
        return 'azure-anthropic' if _has_azure_anthropic_config() else 'anthropic'
    return _normalize_ollama_base_url()


def _ensure_llm_models_db() -> Path:
    db_path = _llm_models_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                UNIQUE(model_name, provider)
            )
            """
        )
        con.execute(
            "INSERT OR IGNORE INTO llm_models (model_name, provider) VALUES (?, ?)",
            (DEFAULT_LLM.model_name, DEFAULT_LLM.provider),
        )
        con.commit()
    return db_path


def list_llm_models() -> List[LLM_Model]:
    db_path = _ensure_llm_models_db()
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT provider, model_name
            FROM llm_models
            ORDER BY CASE WHEN provider = 'openai' THEN 0 ELSE 1 END, LOWER(model_name)
            """
        ).fetchall()
    return [
        LLM_Model(
            provider=_normalize_model_provider(provider, model_name),
            model_name=model_name,
        )
        for provider, model_name in rows
    ]


def add_llm_model(model_name: str, provider: str | None = None) -> LLM_Model:
    normalized_name = model_name.strip()
    if not normalized_name:
        raise ValueError('Model name cannot be empty.')
    normalized_provider = _normalize_model_provider(provider, normalized_name)
    db_path = _ensure_llm_models_db()
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT OR IGNORE INTO llm_models (model_name, provider) VALUES (?, ?)",
            (normalized_name, normalized_provider),
        )
        con.commit()
    return LLM_Model(provider=normalized_provider, model_name=normalized_name)


def delete_llm_model(model_name: str, provider: str | None = None) -> bool:
    normalized_name = model_name.strip()
    if not normalized_name:
        raise ValueError('Model name cannot be empty.')
    normalized_provider = provider.strip() if provider else None
    if normalized_provider and is_provider_url(normalized_provider):
        normalized_provider = _normalize_ollama_base_url(normalized_provider)

    db_path = _ensure_llm_models_db()
    with sqlite3.connect(db_path) as con:
        if normalized_provider:
            cur = con.execute(
                "DELETE FROM llm_models WHERE model_name = ? AND provider = ?",
                (normalized_name, normalized_provider),
            )
        else:
            cur = con.execute(
                "DELETE FROM llm_models WHERE model_name = ?",
                (normalized_name,),
            )
        con.commit()
        deleted = cur.rowcount > 0

    # Keep at least one default option available.
    _ensure_llm_models_db()
    return deleted


def resolve_llm_model(model: str | LLM_Model | None) -> LLM_Model:
    if isinstance(model, LLM_Model):
        normalized_model_name = (model.model_name or '').strip()
        if not normalized_model_name:
            return DEFAULT_LLM
        normalized_provider = _normalize_model_provider(model.provider, normalized_model_name)
        return LLM_Model(provider=normalized_provider, model_name=normalized_model_name)

    # Streamlit/session objects may carry model-like instances that are not the
    # exact LLM_Model class identity in this module; normalize by attributes.
    model_name_attr = getattr(model, 'model_name', None)
    provider_attr = getattr(model, 'provider', None)
    if model_name_attr:
        normalized_model_name = str(model_name_attr).strip()
        if not normalized_model_name:
            return DEFAULT_LLM
        normalized_provider = _normalize_model_provider(provider_attr, normalized_model_name)
        return LLM_Model(provider=normalized_provider, model_name=normalized_model_name)

    if not model:
        return DEFAULT_LLM

    normalized_model_name = str(model).strip()
    if not normalized_model_name:
        return DEFAULT_LLM

    db_path = _ensure_llm_models_db()
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT provider, model_name FROM llm_models WHERE model_name = ? ORDER BY id LIMIT 1",
            (normalized_model_name,),
        ).fetchone()
    if row is not None:
        return LLM_Model(
            provider=_normalize_model_provider(row[0], row[1]),
            model_name=row[1],
        )
    return LLM_Model(
        provider=default_provider_for_model(normalized_model_name),
        model_name=normalized_model_name,
    )





#FIXME: PEP8, type safety, clarity

class Motor(object):
    def __init__(self, table_name_filter=None):
        from dotenv import load_dotenv
        logger.debug(f"Current working directory: {os.getcwd()}")
        config_path = os.path.join(os.getcwd(), "./config.env")
        if not load_dotenv(config_path):
            logger.warning(f"Failed to load config.env in this directory {os.getcwd()}. Make sure to have a config.env file with the required configuration or set environment variables directly.")
            load_dotenv()

        self._table_name_filter=table_name_filter
        self.db_chat = self._dbchat_init()
        self.db_source = self._dbtarget_init()
        
        # Initialize sync manager
        self.sync_manager = VectorStoreSyncManager(db_chat=self.db_chat)
        self._vectorstore = None

    def _ensure_sync_manager(self):
        """Initialize sync manager lazily for partially constructed Motor objects."""
        if not hasattr(self, 'sync_manager') or self.sync_manager is None:
            self.sync_manager = VectorStoreSyncManager(db_chat=getattr(self, 'db_chat', None))
            logger.info("Sync manager initialized lazily")
        elif hasattr(self, 'db_chat') and getattr(self.sync_manager, 'db_chat', None) is not self.db_chat:
            # Rebind after DB reconnection so sync uses the active chat DB config.
            self.sync_manager.set_db_chat(self.db_chat)
        return self.sync_manager

    @property
    def error(self):
        return getattr(self, '_error', None)
    
    @property
    def vectorstore(self):
        return getattr(self, '_vectorstore', None)
    
    @vectorstore.setter
    def vectorstore(self, vs):
        """Set vectorstore and connect sync manager to it."""
        self._vectorstore = vs
        if vs is not None:
            self._ensure_sync_manager().set_vectorstore(vs)
            logger.info("VectorStore connected to sync manager")

    @property
    def current_model(self):
        return getattr(self, '_model', None)

    @current_model.setter
    def current_model(self, model):
        resolved_model = resolve_llm_model(model)
        model_name = resolved_model.model_name
        needs_init = (
            self.current_model != model_name
            or not hasattr(self, '_llm_agent')
            or self._llm_agent is None
        )
        if needs_init:
            from langchain_openai import AzureChatOpenAI, ChatOpenAI
            logger.info(f"Changed to model {model_name}")
            self._model = model_name
            self._model_provider = resolved_model.provider
            if is_provider_url(resolved_model.provider):
                ollama_base_url = _normalize_ollama_base_url(resolved_model.provider)
                self._llm_agent = ChatOpenAI(
                    temperature=0,
                    api_key=lambda: "fdsfs",
                    streaming=True,
                    model=model_name,
                    base_url=ollama_base_url,
                )
                logger.info(f"Ollama API base set to {ollama_base_url} for model {model_name}")
            elif resolved_model.provider in {'azure-openai', 'azure'}:
                from pydantic import SecretStr

                azure_endpoint = _ge(
                    'AZURE_FOUNDRY_URL',
                    _ge('AZURE_BASE_URL', _ge('AZURE_OPENAI_ENDPOINT', '')),
                ).strip()
                api_key = _ge(
                    'AZURE_FOUNDRY_API_KEY',
                    _ge('AZURE_API_KEY', _ge('AZURE_OPENAI_API_KEY', '')),
                ).strip()
                api_version = _ge(
                    'AZURE_FOUNDRY_API_VERSION',
                    _ge('AZURE_API_VERSION', _ge('AZURE_OPENAI_API_VERSION', '2025-04-01-preview')),
                ).strip()
                if not azure_endpoint or not api_key:
                    raise ValueError(
                        'Azure provider requires AZURE_FOUNDRY_URL and AZURE_FOUNDRY_API_KEY '
                        'or AZURE_BASE_URL and AZURE_API_KEY '
                        'or AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.'
                    )
                self._llm_agent = AzureChatOpenAI(
                    temperature=0,
                    api_key=SecretStr(api_key),
                    azure_endpoint=azure_endpoint,
                    api_version=api_version,
                    streaming=True,
                    azure_deployment=model_name,
                    model=model_name,
                )
                logger.info(
                    'Azure OpenAI endpoint set to %s for model %s using api version %s',
                    azure_endpoint,
                    model_name,
                    api_version,
                )
            elif resolved_model.provider == 'azure-anthropic':
                self._llm_agent = _AzureAnthropicAdapter(model_name=model_name)
                logger.info(
                    'Azure Anthropic base URL set to %s for model %s',
                    _ge('AZURE_ANTHROPIC_URL', ''),
                    model_name,
                )
            else:
                self._llm_agent = ChatOpenAI(
                    temperature=0,
                    api_key=lambda: _ge("OPENAI_API_KEY", "fdsfs"),
                    streaming=True,
                    model=model_name,
                )

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

    @property
    def is_new_session(self):
        # Check whether the chat history contains only the initial system message(s) and no user messages
        return all(m.type=='system' for m in self._chat_history.messages) and not any(m.type=='user' for m in self._chat_history.messages)

    # public methods
    def new_session(self, username, email, label=None, referenced_session=None):
        import chromadb
        import random
        import pandas as pd
        from kooplexQuery.history import CustomChatHistory
        # from kooplexQuery import history

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        label=label or random.randbytes(16)
        context = ""
        try:
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
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")
            context = ""
        self._chat_history=CustomChatHistory()
        default_instruction = (
            "You are a helpful assistant that translates natural language questions into SQL queries. "
            "Always try to answer the question with a SQL query based on the provided database schema "
            "and data description. If the question is not clear, ask for clarification. Always use the "
            "provided schema and data description to inform your SQL generation. Do not make up any "
            "information about the database that is not included in the schema or data description."
        )

        # Some vLLM/HF chat templates allow at most one system message at the beginning.
        system_parts = []
        if context:
            system_parts.append(context)
        try:
            instruction = self.db_chat.load_knowledge(reference='instruction')
        except Exception as e:
            logger.error(f"Error loading instruction knowledge: {e}")
            instruction = default_instruction
        if instruction:
            system_parts.append(instruction)
        if not system_parts:
            system_parts.append(default_instruction)

        self._chat_history.add_system_message("\n\n".join(system_parts))
        self._plot_history=CustomChatHistory()
        self._plot_history.add_system_message("""You are a helper to create plots.""")
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
        if self.is_new_session:
            self._chat_history.add_user_message(question, metadata={'type': 'example_question'})
            self._chat_history.add_ai_message(sql, metadata={'type': 'example_sql', 'parsed': [Content_Chunk('sql', sql)]})
            self.sql=sql


    async def chat(self, prompt, model_name=DEFAULT_LLM):
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

    async def correct_error(self, error, model_name=DEFAULT_LLM):
        detail=getattr(error, 'orig', None)
        statement=getattr(error, 'statement', None)
        prompt = f"Correct the SQL query\n{statement}\nbecause: {detail}"
        t0=time.time()
        # Ensure llm agent exists even if no prior chat() call happened.
        self.current_model=model_name
        self._chat_history.add_user_message(prompt, metadata={'timestamp': t0, 'model': self.current_model })
        collected=""
        async for chunk in self._llm_agent.astream(prompt):
            if c:=chunk.content:
                collected += c
                yield c
        self._chat_history.add_ai_message(collected, metadata={'type': 'generated', 'duration': time.time()-t0, 'parsed': self._parse_sql(collected) })

    async def plot(self, instruction_prompt, model_name=DEFAULT_LLM):
        if self.df is not None:
            t0=time.time()
            self.current_model=model_name
            self._chat_history.add_user_message(instruction_prompt, metadata={'type': 'plot', 'model': self.current_model})

            # PLOT WITH LLM  FIXME
#             if self.df.shape[0] < 1000:
#                 tmpdf = self.df.copy()
#             else:
#                 tmpdf = self.df.sample(1000, random_state=42)
#             prompt = f"""
# Data: {tmpdf.to_dict()}

# User instructions for plotting: {instruction_prompt}

# * Refer to the data as 'df'
# * If there are multiple plots then use subplots in matplotlib
#             """
            full_prompt = f"""
This is your base code for obtaining the data:
```python
from kooplexQuery_utils.plot_utils import *            
sql_query = '''{self.sql}'''
df = pd.read_sql(text(sql_query), con=engine)
```
Modify the above code according to the user's instructions!

Use plotly!

User instructions for plotting: {instruction_prompt}
"""
            self._plot_history.add_user_message(full_prompt)
            resp = ""
            async for chunk in self._llm_agent.astream(self._plot_history.messages):
                if c:=chunk.content:
                    resp += c
                    yield c
            self._plot_history.add_ai_message(resp)
            # Extract the python code from the response
            try:
                code = resp.split("```python")[1].split("```")[0].strip()
            except:
                code = resp.split("```")[1].split("```")[0].strip()
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

    async def prepare_save(self, model_name=DEFAULT_LLM):
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
        # logger.info(f"Generating THE question with: {prompt}")
        # Ensure llm agent exists even if save flow starts before any chat() call.
        self.current_model=model_name

        def _fallback_question():
            messages = list(getattr(self._chat_history, 'messages', []) or [])
            for msg in reversed(messages):
                if getattr(msg, 'type', None) == 'user':
                    content = (getattr(msg, 'content', None) or '').strip()
                    if content:
                        return content
            return f"What question is answered by this SQL query? {self.sql}".strip()

        timeout_seconds = int(_ge('SAVE_PREPARE_TIMEOUT_SECONDS', 45))
        try:
            resp = await asyncio.wait_for(self._llm_agent.ainvoke(prompt), timeout=timeout_seconds)
            collected = (getattr(resp, 'content', '') or '').strip()
            if not collected:
                collected = _fallback_question()
            self._question = collected
            logger.info(f"Generated question: {collected}")
            yield collected
        except Exception as e:
            logger.warning(f"prepare_save failed, using fallback question: {e}")
            collected = _fallback_question()
            self._question = collected
            yield collected


    def execute(self, sql):
        try:
            self._error=None
            self._chat_history.add_user_message(sql, metadata={'type': 'submit_sql'})
            self.sql=sql
            t0=time.time()
            df=self.db_source.query_to_df(sql)
            self.df=df
            # self._chat_history.add_ai_message(df.head().to_string(), metadata={'type': 'dataframe', 'dataframe': df, 'query': sql, 'duration': time.time()-t0})
            # Limit stored text to ~5KB to avoid memory/token explosion
            table_text = df.to_string(max_rows=100, max_cols=20)
            max_len = 5 * 1024
            if len(table_text) > max_len:
                table_text = table_text[:max_len-24] + "\n... [truncated]"
                # FIXME Maybe add an extra sentence noting that the table was truncated and that the full table is available in the dataframe attached to the message metadata
            self._chat_history.add_ai_message(
                table_text,
                metadata={'type': 'dataframe', 'dataframe': df, 'query': sql, 'duration': time.time()-t0}
            )

        except Exception as e:
            self._chat_history.add_ai_message(str(e), metadata={'type': 'error', 'content': e, 'query': sql})
            self._error=e
            self.sql=None

    def save_query(self):
        logger.info(f"Saving query with question: {self.question} and sql: {self.sql}")
        self.db_chat.save_query(self.session_id, self.question, self.sql, question_type = 'user', public = False)
        self._question=None
    
    def save_knowledge(self, reference: str, content: str):
        """
        Save knowledge to database and automatically sync to vectorstore.
        
        Args:
            reference: The reference key for the knowledge (e.g., 'schema', 'instruction')
            content: The knowledge content to save
        """
        logger.info(f"Saving knowledge with reference: {reference}")
        self.db_chat.save_knowledge(reference, content)
        # Automatically sync to vectorstore if available
        self._ensure_sync_manager().sync_knowledge(reference, content)
    
    def validate_and_sync_question(self, question_id: int):
        """
        Validate a question and automatically sync it to vectorstore.
        
        Args:
            question_id: The ID of the question to validate
        """
        logger.info(f"Validating question: {question_id}")
        self.db_chat.validate_question(question_id)
        # Fetch the validated question and sync it
        try:
            from sqlalchemy import text
            with self.db_chat.engine.connect() as con:
                q = text("""
                    SELECT q.content, a.sql 
                    FROM question q 
                    JOIN query a ON q.id = a.question_id 
                    WHERE q.id = :qid
                """)
                result = con.execute(q, {'qid': question_id}).fetchone()
                if result:
                    question_content, sql = result
                    self._ensure_sync_manager().sync_example(question_id, question_content, sql, True)
        except Exception as e:
            logger.error(f"Error syncing validated question {question_id}: {e}")

    # private methods
    def _dbchat_init(self, db_config: dict = {}) -> DBChat:
        if db_config:
            logger.debug(f"Initializing DBChat with provided db_config: {db_config}")
            return DBChat(**db_config, generated_callback=lambda c: None)
        else:
            logger.debug(f"Initializing DBChat with host: {_ge('CHAT_HOST', 'localhost')}, port: {int(_ge('CHAT_PORT', 5432))}, database: {_ge('CHAT_DATABASE', 'sewage')}, schema: {_ge('CHAT_SCHEMA', 'chat')}, user: {_ge('CHAT_USER', 'chat_agent')}")
            return DBChat(
                hostname=_ge('CHAT_HOST', 'localhost'), port=int(_ge('CHAT_PORT', 5432)),
                database=_ge('CHAT_DATABASE', 'sewage'), schema=_ge('CHAT_SCHEMA', 'chat'),
                user=_ge('CHAT_USER', 'chat_agent'),
                password=_ge('CHAT_PASSWORD', ''), generated_callback=lambda c: None
            )

    def _dbtarget_init(self, db_config: dict = {}) -> DBQuery:
        if db_config:
            logger.debug(f"Initializing DBQuery with provided db_config: {db_config}")
            return DBQuery(**db_config, generated_callback=lambda c: None)

        else:
            logger.debug(f"Initializing DBQuery with host: {_ge('DB_HOST', 'localhost')}, port: {int(_ge('DB_PORT', 5432))}, database: {_ge('DB_DATABASE', 'sewage')}, schema: {_ge('DB_SCHEMA', 'distilled')}, user: {_ge('DB_USER', 'reader')}")
            return DBQuery(
                hostname=_ge('DB_HOST', 'localhost'), port=int(_ge('DB_PORT', 5432)),
                database=_ge('DB_DATABASE', 'sewage'), schema=_ge('DB_SCHEMA', 'distilled'),
                user=_ge('DB_USER', 'reader'),
                password=_ge('DB_PASSWORD', ''), type=_ge('DB_TYPE', 'postgres'),  
                url=_ge('DB_URL', '')  
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
