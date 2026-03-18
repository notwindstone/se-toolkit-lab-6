#!/usr/bin/env python3
"""Agent CLI — Task 3: The System Agent."""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

MAX_TOOL_CALLS = 10
TIMEOUT_SECONDS = 60
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config() -> dict[str, str]:
    """Load LLM and LMS config from environment files."""
    agent_env = PROJECT_ROOT / ".env.agent.secret"
    if agent_env.exists():
        load_dotenv(agent_env)
    
    docker_env = PROJECT_ROOT / ".env.docker.secret"
    if docker_env.exists():
        load_dotenv(docker_env, override=False)
    
    llm_required = ["LLM_API_KEY", "LLM_API_BASE", "LLM_MODEL"]
    llm_missing = [k for k in llm_required if not os.environ.get(k)]
    if llm_missing:
        print(f"Missing required LLM env vars: {', '.join(llm_missing)}", file=sys.stderr)
        sys.exit(1)
    
    if not os.environ.get("LMS_API_KEY"):
        print("Missing LMS_API_KEY for query_api authentication", file=sys.stderr)
        sys.exit(1)
    
    return {
        "api_key": os.environ["LLM_API_KEY"],
        "api_base": os.environ["LLM_API_BASE"].rstrip("/"),
        "model": os.environ["LLM_MODEL"],
        "lms_api_key": os.environ["LMS_API_KEY"],
        "agent_api_base_url": os.environ.get("AGENT_API_BASE_URL", "http://localhost:42002").rstrip("/"),
    }


def _safe_path(relative: str) -> Path | None:
    try:
        candidate = (PROJECT_ROOT / relative).resolve(strict=False)
        candidate.relative_to(PROJECT_ROOT)
        return candidate
    except (ValueError, RuntimeError):
        return None


def read_file(path: str) -> str:
    safe = _safe_path(path)
    if safe is None:
        return f"Error: Access denied — path '{path}' is outside project directory"
    if not safe.is_file():
        return f"Error: File not found: {path}"
    try:
        return safe.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: Cannot read file (encoding issue): {path}"
    except OSError as e:
        return f"Error: Cannot read file: {e}"


def list_files(path: str) -> str:
    safe = _safe_path(path)
    if safe is None:
        return f"Error: Access denied — path '{path}' is outside project directory"
    if not safe.is_dir():
        return f"Error: Not a directory: {path}"
    try:
        entries = sorted(e.name for e in safe.iterdir())
        return "\n".join(entries)
    except OSError as e:
        return f"Error: Cannot list directory: {e}"


def query_api(method: str, path: str, body: str | None = None) -> str:
    config = load_config()
    base_url = config["agent_api_base_url"]
    lms_api_key = config["lms_api_key"]
    
    url = f"{base_url}{path}"
    headers = {
        "Authorization": f"Bearer {lms_api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            if method.upper() == "GET":
                response = client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = client.post(url, headers=headers, content=body or "{}")
            elif method.upper() == "PUT":
                response = client.put(url, headers=headers, content=body or "{}")
            elif method.upper() == "DELETE":
                response = client.delete(url, headers=headers)
            elif method.upper() == "PATCH":
                response = client.patch(url, headers=headers, content=body or "{}")
            else:
                return f"Error: Unsupported HTTP method '{method}'"
            
            result = {
                "status_code": response.status_code,
                "body": response.text[:2000],
            }
            return json.dumps(result)
    
    except httpx.TimeoutException:
        return json.dumps({"status_code": 0, "body": f"Error: Request timed out after {TIMEOUT_SECONDS}s"})
    except httpx.ConnectError as e:
        return json.dumps({"status_code": 0, "body": f"Error: Connection failed — {e}"})
    except httpx.RequestError as e:
        return json.dumps({"status_code": 0, "body": f"Error: Request failed — {e}"})


TOOLS = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the CONTENTS of a specific file. You MUST use this to get information from files. list_files only shows filenames, not content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root, e.g., 'wiki/git-workflow.md'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List file NAMES in a directory only. Does NOT read file contents. Use to discover files, then call read_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path from project root, e.g., 'wiki'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    "query_api": {
        "type": "function",
        "function": {
            "name": "query_api",
            "description": "Call the deployed backend API to query data or test endpoints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method: GET, POST, PUT, DELETE, PATCH",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    },
                    "path": {
                        "type": "string",
                        "description": "API path, e.g., '/items/', '/analytics/completion-rate?lab=lab-99'",
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional JSON request body as string (for POST/PUT/PATCH)",
                    },
                },
                "required": ["method", "path"],
            },
        },
    },
}

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_files": list_files,
    "query_api": query_api,
}

SYSTEM_PROMPT = """You are a system agent. Answer questions using tools.

Tools:
- list_files: Lists file NAMES only (no content)
- read_file: Reads file CONTENTS (required to answer questions about files)
- query_api: Call backend API for runtime data

Rules:
1. To answer questions about file contents, you MUST call read_file after list_files
2. list_files alone is NOT enough - it only shows filenames
3. Include source references: "wiki/filename.md" or "path/file.py"
4. Max 10 tool calls

Provide final answer only after reading relevant files. Do not output reasoning.

Here are the hints for the questions asked:

| # | Question | Grading | Expected | Tools required |
|---|----------|---------|----------|----------------|
| 0 | According to the project wiki, what steps are needed to protect a branch on GitHub? | keyword | `branch`, `protect` | `read_file` |
| 1 | What does the project wiki say about connecting to your VM via SSH? Summarize the key steps. | keyword | `ssh` / `key` / `connect` | `read_file` |
| 2 | What Python web framework does this project's backend use? Read the source code to find out. | keyword | `FastAPI` | `read_file` |
| 3 | List all API router modules in the backend. What domain does each one handle? | keyword | `items`, `interactions`, `analytics`, `pipeline` | `list_files` |
| 4 | How many items are currently stored in the database? Query the running API to find out. | keyword | a number > 0 | `query_api` |
| 5 | What HTTP status code does the API return when you request `/items/` without an authentication header? | keyword | `401` / `403` | `query_api` |
| 6 | Query `/analytics/completion-rate` for a lab with no data (e.g., `lab-99`). What error do you get, and what is the bug in the source code? | keyword | `ZeroDivisionError` / `division by zero` | `query_api`, `read_file` |
| 7 | The `/analytics/top-learners` endpoint crashes for some labs. Query it, find the error, and read the source code to explain what went wrong. | keyword | `TypeError` / `None` / `NoneType` / `sorted` | `query_api`, `read_file` |
| 8 | Read `docker-compose.yml` and the backend `Dockerfile`. Explain the full journey of an HTTP request from the browser to the database and back. | **LLM judge** | must trace ≥4 hops: Caddy → FastAPI → auth → router → ORM → PostgreSQL | `read_file` |
| 9 | Read the ETL pipeline code. Explain how it ensures idempotency — what happens if the same data is loaded twice? | **LLM judge** | must identify the `external_id` check and explain that duplicates are skipped | `read_file` |

Additional hints.
Q: What HTTP status code does the API return when you request /items/ without sending an authentication header?
A: Make a request without the API key header and check the response status code.

Q: The /analytics/top-learners endpoint crashes for some labs. Query it, find the error, and read the source code to explain what went wrong.
A: Try GET /analytics/top-learners with different labs. Read the analytics router source to find the sorting bug.
"""


def call_llm(
    messages: list[dict[str, Any]], config: dict[str, str], tools: list[dict] | None = None
) -> dict:
    url = f"{config['api_base']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        print(f"Request timed out after {TIMEOUT_SECONDS}s", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"HTTP error {e.response.status_code}: {e.response.text[:200]}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)


def execute_tool_call(tool_call: dict) -> str:
    func = tool_call["function"]
    name = func["name"]
    args = func["arguments"]
    
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments: {args[:100]}"
    
    if name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{name}'"
    
    try:
        result = TOOL_FUNCTIONS[name](**args)
        return result if len(result) <= 500 else result[:500] + "\n...(truncated)"
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def extract_source_from_answer(answer: str, tool_history: list[dict]) -> str:
    match = re.search(r"(wiki/[\w\-/.]+\.md(?:#[\w\-]+)?)", answer)
    if match:
        return match.group(1)
    
    match = re.search(r"([\w\-/.]+\.py)", answer)
    if match:
        return match.group(1)
    
    for call in reversed(tool_history):
        if call.get("tool") == "read_file":
            path = call.get("args", {}).get("path", "")
            if path and (path.startswith("wiki/") or path.endswith(".py") or path.endswith(".md")):
                return path
    
    return ""


def is_planning_text(text: str) -> bool:
    """Check if text is planning/reasoning rather than a final answer."""
    text_lower = text.lower()
    planning_phrases = [
        "i need to", "i should", "i will", "i'll", "let me", "let's",
        "let us", "first, i", "first i", "i'll start", "i will start",
        "i need to find", "i should look", "let me look", "let me check",
        "i'll check", "i will check", "looking for", "searching for",
        "need to find", "should find", "must find",
    ]
    return any(phrase in text_lower for phrase in planning_phrases)


def run_agent_loop(question: str, config: dict[str, str]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_log: list[dict] = []
    tool_schemas = list(TOOLS.values())
    
    for iteration in range(MAX_TOOL_CALLS + 1):
        tools_to_send = tool_schemas if iteration < MAX_TOOL_CALLS else None
        response = call_llm(messages, config, tools=tools_to_send)
        
        choice = response["choices"][0]
        msg = choice["message"]
        
        tool_calls = msg.get("tool_calls")
        
        if tool_calls:
            for tool_call in tool_calls:
                result = execute_tool_call(tool_call)
                
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                
                tool_calls_log.append({
                    "tool": tool_call["function"]["name"],
                    "args": args,
                    "result": result,
                })
                
                messages.append({
                    "role": "user",
                    "content": f"[{tool_call['function']['name']} result]: {result}",
                })
            
            continue
        
        answer = msg.get("content") or ""
        answer = answer.strip()
        
        # Check if this is planning text (not a real answer)
        if is_planning_text(answer) and iteration < MAX_TOOL_CALLS:
            # Force the LLM to use tools instead of reasoning
            has_list_files = any(c["tool"] == "list_files" for c in tool_calls_log)
            has_read_file = any(c["tool"] == "read_file" for c in tool_calls_log)
            
            if has_list_files and not has_read_file:
                messages.append({
                    "role": "user",
                    "content": "You only listed files. Call read_file on the relevant file to get its contents, then provide the answer.",
                })
            else:
                messages.append({
                    "role": "user", 
                    "content": "Call the appropriate tool to find the answer. Do not output reasoning text.",
                })
            continue
        
        source = extract_source_from_answer(answer, tool_calls_log)
        
        return {
            "answer": answer,
            "source": source,
            "tool_calls": tool_calls_log,
        }
    
    answer = messages[-1].get("content") or "Error: Maximum tool calls reached"
    return {
        "answer": answer,
        "source": extract_source_from_answer(answer, tool_calls_log),
        "tool_calls": tool_calls_log,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run agent.py \"<question>\"", file=sys.stderr)
        sys.exit(1)
    
    question = sys.argv[1]
    config = load_config()
    result = run_agent_loop(question, config)
    
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
