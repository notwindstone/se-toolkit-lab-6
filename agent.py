#!/usr/bin/env python3
"""
System Agent - Task 3

A CLI agent that connects to an LLM and can use tools to:
- Read files from the project repository
- List files in directories
- Query the backend API

Outputs structured JSON with answer, source, and tool_calls.
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import httpx
from openai import OpenAI


# =============================================================================
# Configuration
# =============================================================================

def load_config() -> dict[str, str]:
    """Load configuration from environment files and variables."""
    # Load .env.agent.secret for LLM config
    agent_env = Path(".env.agent.secret")
    if agent_env.exists():
        load_dotenv(agent_env)
    
    # Load .env.docker.secret for LMS API key
    docker_env = Path(".env.docker.secret")
    if docker_env.exists():
        load_dotenv(docker_env, override=True)
    
    return {
        "llm_api_key": os.environ.get("LLM_API_KEY", ""),
        "llm_api_base": os.environ.get("LLM_API_BASE", ""),
        "llm_model": os.environ.get("LLM_MODEL", "qwen3-coder-plus"),
        "lms_api_key": os.environ.get("LMS_API_KEY", ""),
        "agent_api_base_url": os.environ.get("AGENT_API_BASE_URL", "http://localhost:42002"),
    }


# =============================================================================
# Tools
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.resolve()


def is_safe_path(path: str) -> bool:
    """Check if a path is safe (no directory traversal outside project)."""
    project_root = get_project_root()
    try:
        # Resolve the full path
        full_path = (project_root / path).resolve()
        # Check if it\'s within the project root
        return str(full_path).startswith(str(project_root))
    except (ValueError, OSError):
        return False


def read_file(path: str) -> dict[str, Any]:
    """
    Read a file from the project repository.
    
    Args:
        path: Relative path from project root.
    
    Returns:
        dict with \'content\' or \'error\' key.
    """
    if not path:
        return {"error": "Path cannot be empty"}
    
    if not is_safe_path(path):
        return {"error": f"Access denied: path \'{path}\' is outside project directory"}
    
    project_root = get_project_root()
    file_path = project_root / path
    
    if not file_path.exists():
        return {"error": f"File not found: {path}"}
    
    if not file_path.is_file():
        return {"error": f"Not a file: {path}"}
    
    # Limit file size to prevent huge responses
    max_size = 100 * 1024  # 100KB
    try:
        content = file_path.read_text(encoding="utf-8")
        if len(content) > max_size:
            content = content[:max_size] + "\\n\\n[... content truncated ...]"
        return {"content": content}
    except UnicodeDecodeError:
        return {"error": f"Cannot read file (not UTF-8): {path}"}
    except PermissionError:
        return {"error": f"Permission denied: {path}"}


def list_files(path: str) -> dict[str, Any]:
    """
    List files and directories at a given path.
    
    Args:
        path: Relative directory path from project root.
    
    Returns:
        dict with \'files\' (newline-separated) or \'error\' key.
    """
    if not path:
        path = "."
    
    if not is_safe_path(path):
        return {"error": f"Access denied: path \'{path}\' is outside project directory"}
    
    project_root = get_project_root()
    dir_path = project_root / path
    
    if not dir_path.exists():
        return {"error": f"Directory not found: {path}"}
    
    if not dir_path.is_dir():
        return {"error": f"Not a directory: {path}"}
    
    try:
        entries = []
        for entry in sorted(dir_path.iterdir()):
            suffix = "/" if entry.is_dir() else ""
            entries.append(f"{entry.name}{suffix}")
        return {"files": "\\n".join(entries)}
    except PermissionError:
        return {"error": f"Permission denied: {path}"}


def query_api(method: str, path: str, body: str | None = None) -> dict[str, Any]:
    """
    Call the deployed backend API.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        path: API path (e.g., /items/)
        body: Optional JSON request body
    
    Returns:
        dict with \'status_code\' and \'body\' keys.
    """
    config = load_config()
    base_url = config["agent_api_base_url"].rstrip("/")
    api_key = config["lms_api_key"]
    
    # Build URL
    url = f"{base_url}{path}"
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Prepare request
    try:
        with httpx.Client(timeout=30.0) as client:
            if method.upper() == "GET":
                response = client.get(url, headers=headers)
            elif method.upper() == "POST":
                json_body = json.loads(body) if body else None
                response = client.post(url, headers=headers, json=json_body)
            elif method.upper() == "PUT":
                json_body = json.loads(body) if body else None
                response = client.put(url, headers=headers, json=json_body)
            elif method.upper() == "DELETE":
                response = client.delete(url, headers=headers)
            elif method.upper() == "PATCH":
                json_body = json.loads(body) if body else None
                response = client.patch(url, headers=headers, json=json_body)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
        
        return {
            "status_code": response.status_code,
            "body": response.text,
        }
    except httpx.RequestError as e:
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON body: {str(e)}"}


# =============================================================================
# Tool Definitions for LLM
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the project repository. Use this to read documentation, source code, or configuration files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root (e.g., \'wiki/git-workflow.md\', \'backend/main.py\')",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a given path. Use this to discover what files exist in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path from project root (e.g., \'wiki\', \'backend\'). Use \'.\' for project root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_api",
            "description": "Call the deployed backend API. Use this to query data from the running system, check API status codes, or test endpoints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method (GET, POST, PUT, DELETE, PATCH)",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    },
                    "path": {
                        "type": "string",
                        "description": "API path (e.g., \'/items/\', \'/analytics/completion-rate\')",
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional JSON request body as a string (for POST, PUT, PATCH)",
                    },
                },
                "required": ["method", "path"],
            },
        },
    },
]

TOOL_MAP = {
    "read_file": read_file,
    "list_files": list_files,
    "query_api": query_api,
}


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a helpful documentation and system agent for a software project.

You have access to three tools:
1. **read_file** - Read files from the project repository (documentation, source code, configs)
2. **list_files** - List files and directories to discover what exists
3. **query_api** - Query the running backend API to get live data or test endpoints

## Guidelines:

- For questions about documentation, git workflow, or project guidelines: use `read_file` on wiki files
- For questions about source code, frameworks, or implementation: use `read_file` on source files or `list_files` to discover modules
- For questions about live data, API behavior, status codes, or database contents: use `query_api`
- Always use tools to find accurate information - don\'t guess
- When you find an answer, include the source (file path with section anchor if applicable)
- If a tool returns an error, try a different approach or path
- Maximum 10 tool calls per question

## Source Format:

When providing a source reference, use: `path/to/file.md#section-anchor`
For API queries, use: `API: /endpoint/path`
For source code, use: `path/to/file.py`

Think step by step. First discover what files exist, then read the relevant ones.
"""


# =============================================================================
# Agent Logic
# =============================================================================

def execute_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Execute a tool and return the result as a string."""
    if tool_name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    try:
        result = TOOL_MAP[tool_name](**args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_agent(question: str, config: dict[str, str]) -> dict[str, Any]:
    """
    Run the agentic loop.
    
    Args:
        question: The user\'s question
        config: Configuration dictionary
    
    Returns:
        dict with answer, source, and tool_calls
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key=config["llm_api_key"],
        base_url=config["llm_api_base"],
    )
    
    # Build message history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    
    # Track tool calls for output
    tool_calls_log = []
    max_iterations = 10
    answer = ""
    source = ""
    
    for iteration in range(max_iterations):
        try:
            # Call LLM
            response = client.chat.completions.create(
                model=config["llm_model"],
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                timeout=50,
            )
            
            msg = response.choices[0].message
            
            # Check for tool calls
            if msg.tool_calls:
                # Log and execute each tool call
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    
                    result = execute_tool(tool_name, args)
                    
                    tool_calls_log.append({
                        "tool": tool_name,
                        "args": args,
                        "result": result,
                    })
                    
                    # Add assistant message with tool call
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_call.function.arguments,
                            },
                        }],
                    })
                    
                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                
                # Continue loop to get LLM response with tool results
                continue
            
            # No tool calls - we have the final answer
            answer = msg.content or ""
            
            # Extract source from answer if present (look for source: pattern)
            source_match = re.search(r"source[:\\s]+([^\\n]+)", answer, re.IGNORECASE)
            if source_match:
                source = source_match.group(1).strip()
            
            break
            
        except Exception as e:
            print(f"Error in iteration {iteration}: {e}", file=sys.stderr)
            if iteration == max_iterations - 1:
                answer = f"Error: {str(e)}"
            break
    
    # If we exhausted iterations without a final answer
    if not answer and tool_calls_log:
        answer = "I was unable to complete the task within the maximum number of tool calls."
    
    return {
        "answer": answer,
        "source": source,
        "tool_calls": tool_calls_log,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="System Agent - Query documentation and backend API using an LLM"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="",
        help="The question to ask the agent",
    )
    
    args = parser.parse_args()
    
    if not args.question:
        print("Error: No question provided", file=sys.stderr)
        print("Usage: uv run agent.py \\"Your question here\\"", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    config = load_config()
    
    # Validate required config
    if not config["llm_api_key"]:
        print("Error: LLM_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    
    if not config["llm_api_base"]:
        print("Error: LLM_API_BASE not set", file=sys.stderr)
        sys.exit(1)
    
    # Run agent
    print(f"Processing question: {args.question}", file=sys.stderr)
    
    result = run_agent(args.question, config)
    
    # Output JSON to stdout
    print(json.dumps(result, ensure_ascii=False))
    
    sys.exit(0)


if __name__ == "__main__":
    main()
