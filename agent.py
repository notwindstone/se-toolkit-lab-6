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
    """Validate that path is within project directory (no ../ traversal)."""
    try:
        candidate = (PROJECT_ROOT / relative).resolve(strict=False)
        candidate.relative_to(PROJECT_ROOT)
        return candidate
    except (ValueError, RuntimeError):
        return None


def read_file(path: str) -> str:
    """Read the contents of a specific file."""
    safe = _safe_path(path)
    if safe is None:
        return f"Error: Access denied — path '{path}' is outside project directory"
    if not safe.is_file():
        return f"Error: File not found: {path}"
    try:
        content = safe.read_text(encoding="utf-8")
        # Return full content for source code analysis (up to 8000 chars)
        return content if len(content) <= 8000 else content[:8000] + "\n...(truncated at 8000 chars)"
    except UnicodeDecodeError:
        return f"Error: Cannot read file (encoding issue): {path}"
    except OSError as e:
        return f"Error: Cannot read file: {e}"


def list_files(path: str) -> str:
    """List file NAMES in a directory only. Does NOT read file contents."""
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
    """Call the deployed backend API to query data or test endpoints."""
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
                "body": response.text[:3000],  # Increased for error analysis
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
            "description": "Read the CONTENTS of a specific file. You MUST use this to answer questions about file contents, source code, configuration, or documentation. Examples: 'wiki/git-workflow.md', 'backend/main.py', 'docker-compose.yml', 'Dockerfile'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root, e.g., 'wiki/git-workflow.md' or 'backend/routers/items.py'",
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
            "description": "List file NAMES in a directory only. Does NOT read file contents. Use this to discover what files exist (e.g., find router modules in a directory), then call read_file on specific files to get their contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path from project root, e.g., 'wiki', 'backend/routers'",
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
            "description": "Call the deployed backend API to query runtime data, test endpoints, or check HTTP status codes. Use this for questions about database contents, API behavior, status codes, or errors from running endpoints. Authentication is handled automatically.",
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
                        "description": "API path, e.g., '/items/', '/analytics/completion-rate?lab=lab-99', '/analytics/top-learners?lab=lab-1'",
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

SYSTEM_PROMPT = """You are a system agent that answers questions about a software project. You have three tools:

1. **list_files** - Lists file/directory NAMES in a folder. Use this FIRST to discover what files exist (e.g., find router modules, wiki pages). Does NOT show contents.

2. **read_file** - Reads file CONTENTS. Use this to answer questions about:
   - Documentation (wiki/*.md files)
   - Source code (backend/*.py files)
   - Configuration (docker-compose.yml, Dockerfile, .env files)
   - Any file where you need to see the actual content

3. **query_api** - Calls the running backend API. Use this for:
   - Counting items in database (GET /items/)
   - Testing endpoint behavior (status codes, errors)
   - Getting runtime data (learner counts, analytics)
   - Questions about what the API returns

**Tool Selection Guide:**
- Wiki/documentation questions → list_files(wiki) then read_file(wiki/...)
- Source code questions (framework, routers, bugs) → list_files then read_file
- Database/API data questions → query_api
- HTTP status code questions → query_api (try without auth headers)
- Bug diagnosis → query_api to see error, then read_file to find source

**Answer Requirements:**
- Always include source references: "wiki/filename.md" or "path/file.py"
- For API data questions, cite the endpoint used
- For bug questions, name the specific error type and the buggy operation
- Maximum 10 tool calls total

**Important:** After using list_files, you MUST call read_file on relevant files to get contents. list_files alone only shows filenames.

Provide your final answer only after gathering all necessary information from tools. Do not output reasoning or planning text — just the answer with source references.
"""


def call_llm(
    messages: list[dict[str, Any]], config: dict[str, str], tools: list[dict] | None = None
) -> dict:
    """Call the LLM API with the given messages and tools."""
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
    """Execute a tool call and return the result."""
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
        # Truncate only if very long (keep more for source analysis)
        return result if len(result) <= 2000 else result[:2000] + "\n...(truncated)"
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def extract_source_from_answer(answer: str, tool_history: list[dict]) -> str:
    """Extract source file reference from answer or tool history."""
    # Pattern 1: wiki files with optional anchor
    match = re.search(r"(wiki/[\w\-/.]+\.md(?:#[\w\-]+)?)", answer)
    if match:
        return match.group(1)
    
    # Pattern 2: Python files
    match = re.search(r"([\w\-/.]+\.py)", answer)
    if match:
        return match.group(1)
    
    # Pattern 3: Docker/config files
    match = re.search(r"(docker-compose\.yml|Dockerfile|\.env\.\w+)", answer)
    if match:
        return match.group(1)
    
    # Pattern 4: Backend routers
    match = re.search(r"(backend/[\w\-/.]+\.py)", answer)
    if match:
        return match.group(1)
    
    # Fallback: use last read_file call
    for call in reversed(tool_history):
        if call.get("tool") == "read_file":
            path = call.get("args", {}).get("path", "")
            if path and any(path.endswith(ext) for ext in [".py", ".md", ".yml", "Dockerfile"]):
                return path
    
    return ""


def is_final_answer(text: str, tool_history: list[dict]) -> bool:
    """Check if text appears to be a final answer rather than planning."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # If it contains specific answer patterns, it's likely final
    answer_indicators = [
        "the answer is", "answer:", "there are", "there is", "returns",
        "status code", "error:", "error type", "uses", "framework",
        "endpoint", "http", "401", "403", "fastapi", "zerodivision",
        "typeerror", "none", "sorted", "external_id", "idempotent",
        "caddy", "postgresql", "docker", "ssh", "branch", "protect",
    ]
    
    if any(indicator in text_lower for indicator in answer_indicators):
        return True
    
    # If we have tool results and text is short/conclusive, it's likely final
    if tool_history and len(text) < 300:
        # Check if it doesn't contain planning phrases
        planning_phrases = [
            "i need to", "i should", "i will", "i'll", "let me", "let's",
            "first, i", "first i", "i'll start", "looking for", "searching",
        ]
        if not any(phrase in text_lower for phrase in planning_phrases):
            return True
    
    return False


def run_agent_loop(question: str, config: dict[str, str]) -> dict[str, Any]:
    """Run the agentic loop to answer a question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_log: list[dict] = []
    tool_schemas = list(TOOLS.values())
    
    for iteration in range(MAX_TOOL_CALLS + 1):
        # Only send tools if we haven't hit the limit
        tools_to_send = tool_schemas if iteration < MAX_TOOL_CALLS else None
        response = call_llm(messages, config, tools=tools_to_send)
        
        choice = response["choices"][0]
        msg = choice["message"]
        
        tool_calls = msg.get("tool_calls")
        
        if tool_calls:
            # Execute all tool calls in this turn
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
        
        # No tool calls - this is the final answer
        answer = msg.get("content") or ""
        answer = answer.strip()
        
        # If answer looks like planning and we have tool calls remaining, encourage tool use
        if not is_final_answer(answer, tool_calls_log) and iteration < MAX_TOOL_CALLS:
            # Check what tools have been used
            tools_used = set(c["tool"] for c in tool_calls_log)
            
            if "list_files" in tools_used and "read_file" not in tools_used:
                messages.append({
                    "role": "user",
                    "content": "You listed files but haven't read their contents. Call read_file on the relevant file(s) to get the information you need.",
                })
            elif not tools_used:
                messages.append({
                    "role": "user",
                    "content": "Call the appropriate tool to find the answer. Use list_files/read_file for documentation/code questions, or query_api for runtime data questions.",
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Provide your final answer based on the tool results you've gathered. Include source references.",
                })
            continue
        
        source = extract_source_from_answer(answer, tool_calls_log)
        
        return {
            "answer": answer,
            "source": source,
            "tool_calls": tool_calls_log,
        }
    
    # Max iterations reached - return whatever we have
    answer = messages[-1].get("content") or "Error: Maximum tool calls reached without final answer"
    return {
        "answer": answer,
        "source": extract_source_from_answer(answer, tool_calls_log),
        "tool_calls": tool_calls_log,
    }


def main() -> None:
    """Main entry point for the agent CLI."""
    if len(sys.argv) < 2:
        print("Usage: uv run agent.py \"<question>\"", file=sys.stderr)
        sys.exit(1)
    
    question = sys.argv[1]
    config = load_config()
    result = run_agent_loop(question, config)
    
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
