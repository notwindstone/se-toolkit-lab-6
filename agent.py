#!/usr/bin/env python3
"""Agent CLI — Task 2: The Documentation Agent.

Outputs JSON with 'answer', 'source', and 'tool_calls' fields.
Implements agentic loop: LLM → tool call → execute → feed back → repeat.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

# Constants
MAX_TOOL_CALLS = 10
TIMEOUT_SECONDS = 60
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config() -> dict[str, str]:
    """Load LLM config from environment variables.
    
    The autochecker injects these directly; .env files are local convenience only.
    """
    required = ["LLM_API_KEY", "LLM_API_BASE", "LLM_MODEL"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    
    return {
        "api_key": os.environ["LLM_API_KEY"],
        "api_base": os.environ["LLM_API_BASE"].rstrip("/"),
        "model": os.environ["LLM_MODEL"],
    }


def _safe_path(relative: str) -> Path | None:
    """Resolve relative path and ensure it stays within project root.
    
    Prevents directory traversal attacks by verifying the resolved path
    is within PROJECT_ROOT.
    
    Returns:
        Absolute Path if valid, None if path traversal attempted.
    """
    try:
        candidate = (PROJECT_ROOT / relative).resolve(strict=False)
        candidate.relative_to(PROJECT_ROOT)
        return candidate
    except (ValueError, RuntimeError):
        return None

def _is_api_runtime_question(question: str) -> bool:
    """Detect questions that require querying the live API."""
    keywords = [
        "status code", "http", "return", "response", "error", 
        "without authentication", "unauthenticated", "bearer",
        "how many", "count", "items", "database", "live", "running"
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


def read_file(path: str) -> str:
    """Read a file from the project repository.
    
    Args:
        path: Relative path from project root (e.g., 'wiki/git-workflow.md')
    
    Returns:
        File contents as string, or error message if file doesn't exist
        or path is invalid.
    """
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
    """List files and directories at a given path.
    
    Args:
        path: Relative directory path from project root (e.g., 'wiki')
    
    Returns:
        Newline-separated listing of entries, or error message.
    """
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
    """Call the backend API with authentication.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        path: API path (e.g., '/items/')
        body: Optional JSON request body
    
    Returns:
        JSON string with status_code and body, or error message.
    """
    api_base = os.environ.get("AGENT_API_BASE_URL", "http://localhost:42002")
    lms_api_key = os.environ.get("LMS_API_KEY")
    
    if not lms_api_key:
        return "Error: LMS_API_KEY not set in environment"
    
    url = f"{api_base.rstrip('/')}{path}"
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
            else:
                return f"Error: Unsupported method '{method}'"
            
            return json.dumps({
                "status_code": response.status_code,
                "body": response.text[:2000],  # Truncate for token limits
            })
    except httpx.TimeoutException:
        return f"Error: Request timed out after {TIMEOUT_SECONDS}s"
    except httpx.RequestError as e:
        return f"Error: Request failed: {e}"

# Tool schemas for LLM function calling
TOOLS = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the project repository. Use to find answers in documentation or source code.",
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
            "description": "List files and directories at a given path. Use to discover available files.",
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
            "description": "Call the backend LMS API to discover runtime behavior. USE THIS when the question asks what the API returns, what status code it gives, or how it behaves at runtime. Examples: 'What status code for unauthenticated request?', 'How many items in database?', 'What error for invalid lab?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method: GET, POST, PUT, DELETE",
                        "enum": ["GET", "POST", "PUT", "DELETE"]
                    },
                    "path": {
                        "type": "string", 
                        "description": "API path with query string if needed, e.g., '/items/' or '/analytics/completion-rate?lab=lab-99'"
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional JSON request body for POST/PUT (as raw JSON string)"
                    }
                },
                "required": ["method", "path"],
            },
        },
    },
}

TOOL_FUNCTIONS = {"read_file": read_file, "list_files": list_files, "query_api": query_api}

SYSTEM_PROMPT = """You are a documentation and system agent for a software engineering course.
Answer questions by reading files OR querying the live API, depending on what the question asks.

TOOL SELECTION RULES (CRITICAL):
1. Use `query_api` when the question asks about:
   - What the API returns (status codes, response bodies, error messages)
   - Runtime behavior (e.g., "what happens when...", "what error do you get")
   - Live data (e.g., "how many items", "what is the score")
   - Endpoint behavior under specific conditions (e.g., "without authentication")

2. Use `read_file` and `list_files` when the question asks about:
   - Source code structure, file contents, or configuration
   - Documentation in the wiki/ directory
   - What a function does, what a config file contains, how something is implemented

3. NEVER answer a runtime behavior question by reading source code — you must use query_api.

4. After using query_api, if you get an error response, you MAY then use read_file 
   to examine the source code and explain the bug.

5. Always include the source reference in your final answer:
   - For wiki/docs: "wiki/filename.md#section-anchor"
   - For API questions: "API response from GET /path"
   - For source code: "backend/app/filename.py#line"

6. Maximum 10 tool calls per question.

Format your final answer as plain text. Do not include JSON or markdown in your response.
"""


def call_llm(
    messages: list[dict[str, Any]], config: dict[str, str], tools: list[dict] | None = None
) -> dict:
    """Call the LLM API and return the parsed response.
    
    Args:
        messages: List of message dicts for the chat completion.
        config: Dict with api_key, api_base, model.
        tools: Optional list of tool schemas for function calling.
    
    Returns:
        Parsed JSON response from the LLM API.
    
    Raises:
        SystemExit on any error (with message to stderr).
    """
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
    """Execute a tool call and return the result as a string.
    
    Args:
        tool_call: Dict with 'function' key containing 'name' and 'arguments'.
    
    Returns:
        Tool execution result as string (truncated to 2000 chars for output).
    """
    func = tool_call["function"]
    name = func["name"]
    args = func["arguments"]
    
    # Parse arguments if they're a JSON string
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments: {args[:100]}"
    
    if name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{name}'"
    
    try:
        result = TOOL_FUNCTIONS[name](**args)
        # Truncate long results to avoid token limits
        return result if len(result) <= 2000 else result[:2999] + "\n...(truncated)"
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def extract_source_from_answer(answer: str, tool_history: list[dict]) -> str:
    """Extract or infer source reference from answer and tool history."""
    # Pattern for wiki files
    match = re.search(r"(wiki/[\w\-/.]+\.md(?:#[\w\-]+)?)", answer)
    if match:
        return match.group(1)
    
    # Pattern for source code files
    match = re.search(r"(backend/[\w\-/.]+\.py)", answer)
    if match:
        return match.group(1)
    
    # Pattern for config files
    match = re.search(r"(pyproject\.toml|docker-compose\.yml|Dockerfile|Caddyfile)", answer)
    if match:
        return match.group(1)
    
    # Fallback: use last read_file path
    for call in reversed(tool_history):
        if call.get("tool") == "read_file":
            path = call.get("args", {}).get("path", "")
            if path and not path.startswith("Error:"):
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
    
    # Track if we've read any files yet
    has_read_file = False

    if _is_api_runtime_question(question):
        hint = "\n\n[HINT: This question asks about API runtime behavior. Use query_api to actually call the endpoint and observe the response.]"
        messages.append({"role": "user", "content": hint})
    
    for iteration in range(MAX_TOOL_CALLS + 1):
        # Force tool usage if we haven't read a file yet and the question requires it
        needs_file_read = any(kw in question.lower() for kw in ["read", "source code", "wiki", "framework", "use", "implement"])
        tools_to_send = tool_schemas if (iteration < MAX_TOOL_CALLS and (not has_read_file or needs_file_read)) else None
        
        response = call_llm(messages, config, tools=tools_to_send)
        choice = response["choices"][0]
        msg = choice["message"]
        
        # Check for tool calls
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                result = execute_tool_call(tool_call)
                
                # Track if we read a file
                if tool_call["function"]["name"] == "read_file":
                    has_read_file = True
                
                # Log the tool call
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls_log.append({
                    "tool": tool_call["function"]["name"],
                    "args": args,
                    "result": result[:500] + "..." if len(result) > 500 else result,  # Truncate for logging
                })
                
                # Append tool result to messages
                messages.append({
                    "role": "user",
                    "content": f"[{tool_call['function']['name']} result]: {result}",
                })
            continue  # Continue loop to get next LLM response
        
        # Final answer reached
        answer = msg.get("content") or ""
        answer = answer.strip()
        
        # If we haven't read a file but the question requires it, force another iteration
        if not has_read_file and needs_file_read and iteration < MAX_TOOL_CALLS:
            messages.append({
                "role": "user",
                "content": "You must read at least one file before answering. Use read_file to examine the relevant file contents.",
            })
            continue
        
        source = extract_source_from_answer(answer, tool_calls_log)
        return {
            "answer": answer,
            "source": source,
            "tool_calls": tool_calls_log,
        }
    
    # Max iterations reached
    answer = messages[-1].get("content", "Error: Maximum tool calls reached") or ""
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
    
    # Output valid JSON to stdout
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
