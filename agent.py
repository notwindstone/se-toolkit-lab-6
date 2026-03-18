#!/usr/bin/env python3
"""Agent CLI — Task 3: The System Agent.

Outputs JSON with 'answer', 'source', and 'tool_calls' fields.
Implements agentic loop: LLM → tool call → execute → feed back → repeat.
Adds query_api tool to interact with the deployed backend API.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# Constants
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
    """Resolve relative path and ensure it stays within project root."""
    try:
        candidate = (PROJECT_ROOT / relative).resolve(strict=False)
        candidate.relative_to(PROJECT_ROOT)
        return candidate
    except (ValueError, RuntimeError):
        return None


def read_file(path: str) -> str:
    """Read a file from the project repository."""
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
    """List files and directories at a given path."""
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
    """Call the deployed backend API."""
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


# Tool schemas for LLM function calling
TOOLS = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the CONTENTS of a specific file. REQUIRED to answer questions about documentation, source code, or configs. You MUST call this after list_files to get actual content. Example: path='wiki/git-workflow.md'",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root, e.g., 'wiki/git-workflow.md' or 'backend/main.py'",
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
            "description": "List file NAMES in a directory only. Does NOT read file contents. Use this first to discover available files, then use read_file to get contents. Example: path='wiki'",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path from project root, e.g., 'wiki' or 'backend/routers'",
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
            "description": "Call the deployed backend API to query data, check status codes, or test endpoints. Use for questions about runtime data, API behavior, or debugging errors.",
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

SYSTEM_PROMPT = """You are a system agent for a software engineering course project.
Answer questions by reading files (documentation, source code) and querying the deployed backend API.

Available tools:
1. list_files — Lists file NAMES in a directory. Does NOT read contents. Use first to discover files.
2. read_file — Reads file CONTENTS. You MUST use this to get actual information from files.
3. query_api — Call the backend API (GET/POST/etc.) to query data or test endpoints.

CRITICAL RULES:
- list_files ONLY returns filenames, NOT content. You CANNOT answer questions using only list_files.
- You MUST call read_file on relevant files to get their contents before answering.
- For wiki/documentation questions: (1) list_files to find files, (2) read_file on relevant files, (3) answer with source
- For source code questions: read_file on relevant .py files, then answer
- For runtime data/API questions: use query_api
- Always include source references: "wiki/filename.md" or "path/to/file.py"
- Maximum 10 tool calls per question

NEVER answer a question about file contents without calling read_file first.
"""


def call_llm(
    messages: list[dict[str, Any]], config: dict[str, str], tools: list[dict] | None = None
) -> dict:
    """Call the LLM API and return the parsed response."""
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
    """Execute a tool call and return the result as a string."""
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
    """Extract or infer source reference from answer and tool history."""
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


def run_agent_loop(question: str, config: dict[str, str]) -> dict[str, Any]:
    """Run the agentic loop and return the final output."""
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
        
        # Final answer (no tool calls)
        answer = msg.get("content") or ""
        answer = answer.strip()
        
        # Check if agent only used list_files without read_file for file-related questions
        has_list_files = any(c["tool"] == "list_files" for c in tool_calls_log)
        has_read_file = any(c["tool"] == "read_file" for c in tool_calls_log)
        has_query_api = any(c["tool"] == "query_api" for c in tool_calls_log)
        
        # If only list_files was used but question is about file contents, force read_file
        if has_list_files and not has_read_file and not has_query_api:
            messages.append({
                "role": "user",
                "content": "You only listed files but did not read their contents. You MUST call read_file on the relevant file(s) to get the actual content before answering. Please call read_file now.",
            })
            continue
        
        source = extract_source_from_answer(answer, tool_calls_log)
        
        return {
            "answer": answer,
            "source": source,
            "tool_calls": tool_calls_log,
        }
    
    # Max iterations reached
    answer = messages[-1].get("content") or "Error: Maximum tool calls reached"
    return {
        "answer": answer,
        "source": extract_source_from_answer(answer, tool_calls_log),
        "tool_calls": tool_calls_log,
    }


def main() -> None:
    """Entry point."""
    if len(sys.argv) < 2:
        print("Usage: uv run agent.py \"<question>\"", file=sys.stderr)
        sys.exit(1)
    
    question = sys.argv[1]
    config = load_config()
    result = run_agent_loop(question, config)
    
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
