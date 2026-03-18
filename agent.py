#!/usr/bin/env python3
"""Agent CLI — Task 3: The System Agent with qwen3-coder-plus."""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
    from dotenv import load_dotenv
except ImportError as e:
    print(json.dumps({"answer": f"Import error: {e}", "source": "", "tool_calls": []}), file=sys.stdout)
    sys.exit(0)

MAX_TOOL_CALLS = 10
TIMEOUT_SECONDS = 60
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config() -> Dict[str, str]:
    """Load LLM and LMS config from environment files."""
    # Try to load from .env files (local development)
    agent_env = PROJECT_ROOT / ".env.agent.secret"
    if agent_env.exists():
        load_dotenv(agent_env)
    
    docker_env = PROJECT_ROOT / ".env.docker.secret"
    if docker_env.exists():
        load_dotenv(docker_env, override=False)

    # Check required variables
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_api_base = os.environ.get("LLM_API_BASE", "")
    llm_model = os.environ.get("LLM_MODEL", "")
    lms_api_key = os.environ.get("LMS_API_KEY", "")
    
    if not llm_api_key or not llm_api_base or not llm_model:
        # Return minimal config - agent will fail gracefully
        return {
            "api_key": llm_api_key,
            "api_base": llm_api_base,
            "model": llm_model,
            "lms_api_key": lms_api_key,
            "agent_api_base_url": os.environ.get("AGENT_API_BASE_URL", "http://localhost:42002").rstrip("/"),
        }

    return {
        "api_key": llm_api_key,
        "api_base": llm_api_base.rstrip("/"),
        "model": llm_model,
        "lms_api_key": lms_api_key,
        "agent_api_base_url": os.environ.get("AGENT_API_BASE_URL", "http://localhost:42002").rstrip("/"),
    }


def _safe_path(relative: str) -> Optional[Path]:
    """Validate that a path is within the project directory."""
    try:
        candidate = (PROJECT_ROOT / relative).resolve(strict=False)
        candidate.relative_to(PROJECT_ROOT)
        return candidate
    except (ValueError, RuntimeError):
        return None


def read_file(path: str) -> str:
    """Read the contents of a file with security checks."""
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
    """List files in a directory with security checks."""
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


def query_api(method: str, path: str, body: Optional[str] = None) -> str:
    """Call the deployed backend API with authentication."""
    try:
        config = load_config()
        base_url = config["agent_api_base_url"]
        lms_api_key = config["lms_api_key"]
        url = f"{base_url}{path}"
        headers = {
            "Authorization": f"Bearer {lms_api_key}",
            "Content-Type": "application/json",
        }
        
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            method_upper = method.upper()
            if method_upper == "GET":
                response = client.get(url, headers=headers)
            elif method_upper == "POST":
                response = client.post(url, headers=headers, content=body or "{}")
            elif method_upper == "PUT":
                response = client.put(url, headers=headers, content=body or "{}")
            elif method_upper == "DELETE":
                response = client.delete(url, headers=headers)
            elif method_upper == "PATCH":
                response = client.patch(url, headers=headers, content=body or "{}")
            else:
                return json.dumps({"status_code": 0, "body": f"Error: Unsupported HTTP method '{method}'"})
            
            result = {
                "status_code": response.status_code,
                "body": response.text,
            }
            return json.dumps(result)
    except httpx.TimeoutException:
        return json.dumps({"status_code": 0, "body": f"Error: Request timed out after {TIMEOUT_SECONDS}s"})
    except httpx.ConnectError as e:
        return json.dumps({"status_code": 0, "body": f"Error: Connection failed — {e}"})
    except httpx.RequestError as e:
        return json.dumps({"status_code": 0, "body": f"Error: Request failed — {e}"})
    except Exception as e:
        return json.dumps({"status_code": 0, "body": f"Error: {type(e).__name__}: {e}"})


# Tool schemas for LLM function calling
TOOLS = [
    {
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
    {
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
    {
        "type": "function",
        "function": {
            "name": "query_api",
            "description": "Call the deployed backend API to query data or test endpoints. Use for runtime data questions like item counts, status codes, or endpoint behavior.",
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
]

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_files": list_files,
    "query_api": query_api,
}

SYSTEM_PROMPT = """You are a system agent for a Learning Management System project. Answer questions using tools.

Tools available:
- list_files: Lists file NAMES only (no content). Use to discover files in a directory.
- read_file: Reads file CONTENTS. REQUIRED to answer questions about file contents, code, or documentation.
- query_api: Call the backend API (GET/POST/PUT/DELETE/PATCH) for runtime data like item counts, status codes, or endpoint behavior.

Rules:
1. To answer questions about file contents, you MUST call read_file after list_files discovers the file.
2. list_files alone is NOT enough - it only shows filenames, not content.
3. For API data questions (counts, status codes, errors), use query_api with proper authentication.
4. For bug diagnosis questions: FIRST query the API to see the actual error, THEN read the source code to find the bug.
5. Include source references in your answer: "wiki/filename.md" or "path/file.py" or "API: /endpoint/".
6. Maximum 10 tool calls per question.
7. Provide final answer only after reading relevant files or querying the API.
8. Do not output reasoning text - provide the direct answer.

When to use each tool:
- Wiki/documentation questions → list_files to find file, then read_file to get content
- Source code questions → read_file directly if you know the path
- Runtime data questions (counts, status codes, errors) → query_api
- Bug diagnosis questions → query_api FIRST to see error, THEN read_file to find bug in code
- System architecture questions → read_file on docker-compose.yml, Dockerfile, main.py
"""


def call_llm(
    messages: List[Dict[str, Any]], config: Dict[str, str], tools: Optional[List[Dict]] = None
) -> Dict:
    """Call the LLM API with retry logic."""
    if not config.get("api_key") or not config.get("api_base"):
        raise ValueError("Missing LLM configuration")
    
    url = f"{config['api_base']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def execute_tool_call(tool_call: Dict) -> str:
    """Execute a tool call and return the result."""
    try:
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

        result = TOOL_FUNCTIONS[name](**args)
        return result
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def extract_source_from_answer(answer: str, tool_history: List[Dict]) -> str:
    """Extract source reference from answer or tool history."""
    match = re.search(r"(wiki/[\w\-/.]+\.md(?:#[\w\-]+)?)", answer)
    if match:
        return match.group(1)
    match = re.search(r"([\w\-/.]+\.py)", answer)
    if match:
        return match.group(1)
    match = re.search(r"([\w\-/.]+\.yml)", answer)
    if match:
        return match.group(1)

    for call in reversed(tool_history):
        if call.get("tool") == "read_file":
            path = call.get("args", {}).get("path", "")
            if path and (path.startswith("wiki/") or path.endswith(".py") or path.endswith(".md") or path.endswith(".yml")):
                return path
        elif call.get("tool") == "query_api":
            path = call.get("args", {}).get("path", "")
            if path:
                return f"API: {path}"

    return ""


def is_planning_text(text: str) -> bool:
    """Check if text is planning/reasoning rather than a final answer."""
    if not text:
        return True
    text_lower = text.lower()
    planning_phrases = [
        "i need to", "i should", "i will", "i'll", "let me", "let's",
        "let us", "first, i", "first i", "i'll start", "i will start",
        "i need to find", "i should look", "let me look", "let me check",
        "i'll check", "i will check", "looking for", "searching for",
        "need to find", "should find", "must find", "i'll use", "i will use",
    ]
    return any(phrase in text_lower for phrase in planning_phrases)


def run_agent_loop(question: str, config: Dict[str, str]) -> Dict[str, Any]:
    """Main agentic loop: LLM calls tools, agent executes, repeat until answer."""
    
    # Check if LLM is configured
    if not config.get("api_key") or not config.get("api_base"):
        return {
            "answer": "Error: LLM not configured. Set LLM_API_KEY, LLM_API_BASE, and LLM_MODEL environment variables.",
            "source": "",
            "tool_calls": [],
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_log: List[Dict] = []

    for iteration in range(MAX_TOOL_CALLS + 1):
        try:
            tools_to_send = TOOLS if iteration < MAX_TOOL_CALLS else None
            response = call_llm(messages, config, tools=tools_to_send)
        except Exception as e:
            # LLM call failed - return error
            return {
                "answer": f"Error: LLM call failed - {type(e).__name__}: {e}",
                "source": "",
                "tool_calls": tool_calls_log,
            }

        try:
            choice = response["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError):
            return {
                "answer": "Error: Invalid LLM response format",
                "source": "",
                "tool_calls": tool_calls_log,
            }

        tool_calls = msg.get("tool_calls")

        if tool_calls:
            # Execute each tool call
            for tool_call in tool_calls:
                try:
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
                        "result": result[:500] + "..." if len(result) > 500 else result,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    })
                except Exception as e:
                    tool_calls_log.append({
                        "tool": "unknown",
                        "args": {},
                        "result": f"Error executing tool: {e}",
                    })
            continue

        # No tool calls - this should be the final answer
        answer = msg.get("content") or ""
        answer = answer.strip()

        # Check if this is planning text (not a real answer)
        if is_planning_text(answer) and iteration < MAX_TOOL_CALLS:
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

        # Extract source and return final answer
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
    """Entry point: parse CLI args, run agent, output JSON."""
    try:
        if len(sys.argv) < 2:
            result = {
                "answer": "Usage: agent.py \"<question>\"",
                "source": "",
                "tool_calls": [],
            }
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(0)

        question = sys.argv[1]
        config = load_config()
        result = run_agent_loop(question, config)

        # Output only valid JSON to stdout
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)
        
    except Exception as e:
        # Catch ALL exceptions and return valid JSON
        result = {
            "answer": f"Error: {type(e).__name__}: {e}",
            "source": "",
            "tool_calls": [],
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)


if __name__ == "__main__":
    main()
