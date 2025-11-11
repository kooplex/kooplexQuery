# client.py
# Author: Jozsef Steger
from __future__ import annotations

import os
import sys
import re
import base64
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("client")

# ---------- Config ----------

RUN_SQL_TOOL = os.getenv("MCP_TOOL_RUN_SQL", "run")
DEFAULT_FETCH = os.getenv("SQL_FETCH", "all")  # "one" or "all"

# Prefer: provide the prompt as a resource (most robust)
PROMPT_RESOURCE_ID = os.getenv("MCP_PROMPT_RESOURCE_ID", "prompts/sql_generation")
# If you still want to try the prompt API:
PROMPT_ID = os.getenv("MCP_PROMPT_ID", "sql_generation")

RESOURCE_ID = os.getenv("MCP_RESOURCE_ID", "doc://viralprimer")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ---------- Helpers ----------
READ_ONLY_PATTERN = re.compile(
    r"\b(UPDATE|DELETE|INSERT|UPSERT|MERGE|TRUNCATE|DROP|ALTER|CREATE|GRANT|REVOKE|VACUUM|ANALYZE)\b",
    re.IGNORECASE,
)



def extract_sql_only(text: str) -> str:
    t = text.strip()
    t = re.sub(r"```sql\s*([\s\S]*?)```", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"```\s*([\s\S]*?)```", r"\1", t)
    t = t.strip("` \n")
    statements = [s.strip() for s in re.split(r";\s*\n?|;\s*$", t) if s.strip()]
    for s in statements:
        if re.match(r"^\s*(SELECT|WITH)\b", s, re.IGNORECASE):
            return s
    return statements[0] if statements else t


def ensure_read_only(sql: str) -> None:
    if READ_ONLY_PATTERN.search(sql):
        raise ValueError("Generated SQL contains non read-only operation.")


# ---------- Ollama ----------
async def call_ollama(model: str, messages: list[dict], tools: list[dict]) -> str:
    """
    Use /api/chat for chat-based models.
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
        "temperature": 0,
        "options": {"num_ctx": 16384},
    }
    full_text = []
    pending_tool_call = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # skip incomplete fragments
                    continue
                # --- Streaming token text ---
                if "message" in data and "content" in data["message"]:
                    token = data["message"]["content"]
                    print(token, end="", flush=True, file=sys.stderr)
                    full_text.append(token)
                # --- Optional: tool call streaming ---
                if "message" in data and "tool_calls" in data["message"]:
                    _tool_trigger=data['message']['tool_calls']
                    log.info("tool trigger: %r", _tool_trigger)
                    #FIXME assumed single trigger
                    _t=_tool_trigger[0]
                    tc=_t['function']
                    pending_tool_call = {
                        "type": "tool_call",
                        "name": tc["name"],
                        "arguments": tc.get("arguments", {}) or {},
                    }
                if data.get("done"):
                    break
    if pending_tool_call:
        return pending_tool_call
    print(file=sys.stderr)
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": "".join(full_text).strip()
        }
    }


# ---------- MCP compatibility shims ----------
class Mcp:
    def __init__(self, session: ClientSession):
        self.s = session

    async def initialize(self) -> None:
        log.debug("Initializing MCP session…")
        await self.s.initialize()
        log.debug("MCP session initialized")


    async def list_tools(self) -> List[Tool]:
        log.info("Calling list tools")
        fn = self.s.list_tools
        result = await fn()
        tools = result.tools
        log.debug("Tools found: " + ", ".join([t.name for t in tools]))
        return tools

    async def call_tool(self, name: str, payload: Dict[str, Any]) -> List[TextContent]:
        log.info("Calling tool %s with %s", name, payload)
        fn = self.s.call_tool
        result = await fn(name, {'payload': payload})
        log.debug("Response: %r", result)
        return result.content

    async def list_resources(self) -> List[Rersource]:
        log.info("Calling list resources")
        fn = self.s.list_resources
        result = await fn()
        resources = result.resources
        log.debug("Resources found: " + ", ".join([r.name for r in resources]))
        return resources

    async def read_resource(self, resource_id: str) -> Optional[TextResourceContents]:
        """Try multiple shapes to read resource text. Return None if not found."""
        log.info("Reading resource: %s", resource_id)
        fn = self.s.read_resource
        try:
            resource = await fn(resource_id)
            return resource.contents[0]
        except Exception as e:
            log.warning("failed: %r", e)
        return None

    async def get_prompt(self, prompt_id: str) -> Optional[PromptMessage]:
        """Try multiple APIs to fetch a prompt. Return None if unsupported."""
        log.info("Fetching prompt: %s", prompt_id)
        fn = self.s.get_prompt
        try:
            prompt = await fn(prompt_id)
            log.debug("prompt %s", prompt)
            return prompt.messages[0]
        except Exception as e:
            log.warning("failed: %r", e)
            raise
        return None


# building blocks
def _par():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH','')}"
    return StdioServerParameters(
        command="python",
        args=["-u", "-m", "mcp_server.entry"],
        cwd=str(repo_root),
        env=env,
    )


async def prompt() -> PromptMessage:
    params = _par()
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            compat = Mcp(session)
            # Initialize
            try:
                await compat.initialize()
            except Exception as e:
                log.error("initialize() failed: %r", e)
                raise
            prompt = await compat.get_prompt(PROMPT_ID)
            if not prompt:
                log.error("prompt %r not found", PROMPT_ID)
                raise Exception("prompt %r not found", PROMPT_ID)
            log.info("Using prompt %r", PROMPT_ID)
            return prompt


async def tools() -> List[Tool]:
    params = _par()
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            compat = Mcp(session)
            # Initialize
            try:
                await compat.initialize()
            except Exception as e:
                log.error("initialize() failed: %r", e)
                raise
            tools = await compat.list_tools()
            if not tools:
                log.error("tools not found")
                raise Exception("tools not found")
            log.debug("tools %r", tools)
            return tools


async def resources() -> List[Resource]:
    params = _par()
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            compat = Mcp(session)
            # Initialize
            try:
                await compat.initialize()
            except Exception as e:
                log.error("initialize() failed: %r", e)
                raise
            resources = await compat.list_resources()
            if not resources:
                log.error("resorces not found")
                raise Exception("resources not found")
            log.debug("resources %r", resources)
            return resources

# ---------- End-to-end ----------
def _resolve_payload_schema(input_schema: dict) -> dict:
    """Return the JSON Schema for the tool arguments (inside payload if present)."""
    props = input_schema.get("properties", {})
    # Case A: payload with $ref → "#/$defs/Name"
    payload = props.get("payload")
    if isinstance(payload, dict) and "$ref" in payload:
        ref = payload["$ref"]  # e.g. "#/$defs/RunInput"
        parts = ref.split("/")
        # Expect ["#", "$defs", "RunInput"]
        if len(parts) == 3 and parts[1] in input_schema:
            return input_schema[parts[1]].get(parts[2], {})
        # Fallback: try to navigate generically
        target = input_schema
        for p in parts[1:]:
            target = target.get(p, {})
        return target

    # Case B: payload inline (no $ref)
    if "payload" in props:
        inner = props["payload"]
        if isinstance(inner, dict):
            # Might itself be an object schema
            return inner

    # Case C: no payload wrapper — use the input schema itself (minus title etc.)
    return input_schema

def _normalize_properties(schema_obj: dict) -> tuple[dict, list]:
    """Return (properties, required) normalized for Ollama tools."""
    properties = dict(schema_obj.get("properties", {}))  # shallow copy
    required = list(schema_obj.get("required", []))

    # Normalize nullable fields: anyOf [string, null] → type ["string","null"]
    for k, v in properties.items():
        if isinstance(v, dict) and "anyOf" in v:
            types = set()
            new_v = dict(v)
            for alt in v["anyOf"]:
                if "type" in alt:
                    types.add(alt["type"])
            if types:
                new_v.pop("anyOf", None)
                # Convert to array type when nullable
                if "null" in types:
                    types.discard("null")
                    if len(types) == 1:
                        new_v["type"] = [list(types)[0], "null"]
                    else:
                        new_v["type"] = list(types) + ["null"]
                else:
                    # multiple non-null types (rare) — keep first for simplicity
                    new_v["type"] = list(types)[0] if len(types) == 1 else list(types)
                properties[k] = new_v

    return properties, required

async def build_ollama_tools(from_mcp_list_tools) -> list[dict]:
    _tools = []
    ts = await from_mcp_list_tools()

    for t in ts:
        input_schema = t.inputSchema or {}
        arg_schema = _resolve_payload_schema(input_schema)
        props, required = _normalize_properties(arg_schema)

        func = {
            "name": t.name,
            "description": t.description or "",  # ensure string
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,  # optional but helpful
            },
        }
        _tools.append({"type": "function", "function": func})
    return _tools

def _slug(s: str) -> str:
    # simple, predictable function-name slug
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = f"r_{s}"
    return s

async def build_ollama_resources(from_mcp_list_resources) -> Tuple[List[dict], Dict[str, dict]]:
    """
    Returns (ollama_tools_for_resources, registry)
    registry maps tool_name -> {"uri": ..., "name": ..., "description": ...}
    """
    tools_for_ollama = []
    registry: Dict[str, dict] = {}
    rs = await from_mcp_list_resources()
    for r in rs:
        tool_name = f"read_resource__{_slug(r.name)}"  # double underscore to avoid collisions
        # Save lookup info
        registry[tool_name] = {
            "uri": r.uri,
            "name": r.name,
            "description": (r.description or "").strip(),
        }

        func = {
            "name": tool_name,
            "description": (
                f"Fetch the content of MCP resource '{r.name}' "
                f"({r.uri}). {registry[tool_name]['description']}"
            ),
            "parameters": {
                "type": "object",
                "properties": {},   # no args needed
                "required": [],
                "additionalProperties": False,
            },
        }
        tools_for_ollama.append({"type": "function", "function": func})
    return tools_for_ollama, registry


def dump_messages(dumpfile: str, messages: list) -> None:
    with open(dumpfile, 'w') as f:
        t0=None
        for m in messages:
            t=m['timestamp']
            dt=t-t0 if t0 else 0
            t0=t
            r=m['role'].upper()
            if 'tool_calls' in m:
                c='MCP ' + str(m['tool_calls'])
            else:
                c=m['content']
            f.write(f"---{t} ({dt})\n{r}: {c}\n")



# ---------- CLI ----------
def main():
    import argparse
    safe=lambda x: x.replace('%', '%%')

    response = httpx.get(f"{OLLAMA_URL}/api/tags")
    models = {
        k: (x['name'], x['size'])
        for k, x in enumerate(response.json().get("models"), start=1)
    }
    model_help="\n".join(f"\t{k}. {safe(n)} (size: {s})" for k, (n, s) in models.items())

    examples = {
        1: "Which tables contain the word 'viralprimer'?",
        2: "List tables and their columns that contain the word 'viralprimer'?",
        3: "Select 5 example records from all tables that contain the word 'viralprimer'?",
        4: "Summarize what 'viralprimer' is about!",
        5: "Which are the 10 most frequent SARS-CoV-2 nucleotide mutations in GISAID database across all samples?",
        6: "Which are the 10 most frequent SARS-CoV-2 nucleotide mutations in GISAID database across all samples? Make sure you get results from the viralprimer database after constructing a valid SQL query. All table name start with 'viralprimer_server_'. Pay extra attention to column names that look similar and use the most appropriate according to column descriptions.",
        7: "List 10 random rare SARS-CoV-2 mutations (frequency below 0.01%) reported in CoVEO.",
        8: "Select 5 example records from all tables that contain the word 'viralprimer'? Use an appropriate tool to access the database.",
    }
    example_help = "\n".join(f"\t{k}. {safe(v)}" for k, v in examples.items())

    parser = argparse.ArgumentParser(
        description="MCP-aware chat: question → oracle (Ollama) → executor (Ollama + MCP) →  backend (postgre SQL).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "question", 
        nargs="?", default=None, 
        help="User question to translate to SQL and run."
    )
    parser.add_argument(
        "--model", 
        default=1, type=int,
        choices=models.keys(),
        help=f"Select language model to use for reasoning. Default is 1. Choises:\n{model_help}"
    )
    parser.add_argument(
        "--test",
        type=int,
        choices=examples.keys(),
        help=f"Choose an example to ask the oracle:\n{example_help}",
    )
    parser.add_argument(
        "--dump", 
        type=str, required=False,
        help=f"If set dump the whole conversation to a file in the end."
    )
#    parser.add_argument("--repl", action="store_true", help="Interactive mode.")
    args = parser.parse_args()
    if args.test:
        q=examples[args.test]
#parser.print_help(sys.stderr)
    else:
        q=args.question
    model=models[args.model][0]
    

    async def run_once(model: str, user_prompt: str) -> list:
        messages = [{"role": "user", "content": user_prompt, "timestamp": time.time()}]
        try:
            log.debug("listing tools...")
            _tools = await build_ollama_tools(tools)
            log.debug("extending tools with resources...")
            _resources, resource_registry = await build_ollama_resources(resources)
            _tools.extend( _resources )
            loop=True

            while loop:
                resp = await call_ollama(model=model, messages=messages, tools=_tools)
                if resp["type"]=="assistant":
                    m=resp["message"]
                    m["timestamp"]=time.time()
                    messages.append(m)
                    loop=False
                elif resp["type"]=="tool_call":
                    params = _par()
                    async with stdio_client(params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            compat = Mcp(session)
                            try:
                                await compat.initialize()
                            except Exception as e:
                                log.error("initialize() failed: %r", e)
                                raise
                            try:
                                tool_name=resp['name']
                                tool_args=resp['arguments']
                                content_str = None
                                if tool_name in resource_registry:
                                    # read resource
                                    uri = resource_registry[tool_name]["uri"]
                                    rr = await compat.read_resource(uri)
                                    content_str = rr.text
                                else:
                                    # call tool
                                    ct = await compat.call_tool(tool_name, tool_args)
                                    content_str = json.dumps(ct[0].text)
                            except Exception as e:
                                log.critical(e)
                                raise
                            if not content_str:
                                log.error("oops")
                                raise Exception("oops")
                            messages.append({
                                "role": "assistant",
                                "timestamp": time.time(),
                                "content": None,
                                "tool_calls": [{"name": tool_name, "arguments": tool_args}],
                            })
                            messages.append({
                                "role": "tool",
                                "timestamp": time.time(),
                                "name": tool_name,
                                "content": content_str,
                            })
                            log.debug(str(messages))
                else:
                    raise Exception(f"Unhandled response type: {resp['type']}")

        except Exception as e:
            print(f"\n[ERROR] {e}")
        return messages

    messages=asyncio.run(run_once(model=model, user_prompt=q))
    dump_messages(dumpfile=args.dump, messages=messages)
#    if args.repl:
#        print("Type your question (empty to exit):")
#        while True:
#            q = input("> ").strip()
#            if not q:
#                break
#            asyncio.run(run_once(q))
#    else:
#        if not args.question:
#            parser.error("Provide a question or use --repl")
#        asyncio.run(run_once(args.question))

if __name__ == "__main__":
    main()

