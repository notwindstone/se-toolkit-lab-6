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
from dotenv import load_dotenv

# Constants
MAX_TOOL_CALLS = 10
TIMEOUT_SECONDS = 60
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config() -> dict[str, str]:
    """Load LLM config from environment."""
    env_file = PROJECT_ROOT / ".env.agent.secret"
    if env_file.exists():
        load_dotenv(env_file)

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
}

TOOL_FUNCTIONS = {"read_file": read_file, "list_files": list_files}

SYSTEM_PROMPT = """You are a documentation agent for a software engineering course.
Answer questions by reading files in the project wiki (wiki/ directory) and source code.

Rules:
1. Use list_files to discover available files in a directory.
2. Use read_file to read specific files and find answers.
3. Always include the source reference in your final answer: "wiki/filename.md#section-anchor".
4. If you cannot find the answer, say so honestly.
5. Maximum 10 tool calls per question.

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
        Tool execution result as string (truncated to 500 chars for output).
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
        return result if len(result) <= 500 else result[:500] + "\n...(truncated)"
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def extract_source_from_answer(answer: str, tool_history: list[dict]) -> str:
    """Extract or infer source reference from answer and tool history.
    
    Priority:
    1. Regex match for wiki/*.md#anchor pattern in answer text
    2. Last read_file path from tool history if it starts with 'wiki/'
    3. Empty string if no source can be determined
    
    Args:
        answer: The LLM's final answer text.
        tool_history: List of tool calls made during the agentic loop.
    
    Returns:
        Source reference string (e.g., 'wiki/git-workflow.md#section').
    """
    # Pattern: wiki/something.md#anchor or wiki/something.md
    match = re.search(r"(wiki/[\w\-/.]+\.md(?:#[\w\-]+)?)", answer)
    if match:
        return match.group(1)
    
    # Fallback: use last read_file path
    for call in reversed(tool_history):
        if call.get("tool") == "read_file":
            path = call.get("args", {}).get("path", "")
            if path.startswith("wiki/"):
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
        # Don't send tools on last iteration to force final answer
        tools_to_send = tool_schemas if iteration < MAX_TOOL_CALLS else None
        response = call_llm(messages, config, tools=tools_to_send)
        
        choice = response["choices"][0]
        msg = choice["message"]
        
        # Check for tool calls
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                # Execute the tool
                result = execute_tool_call(tool_call)
                
                # Log the tool call for output
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
                
                # Append tool result to messages for LLM context
                # Use "user" role instead of "tool" for broader API compatibility
                messages.append({
                    "role": "user",
                    "content": f"[{tool_call['function']['name']} result]: {result}",
                })
            
            # Continue loop to get next LLM response with tool results
            continue
        
        # Final answer (no tool calls)
        answer = msg.get("content") or ""
        answer = answer.strip()
        source = extract_source_from_answer(answer, tool_calls_log)
        
        return {
            "answer": answer,
            "source": source,
            "tool_calls": tool_calls_log,
        }
    
    # Max iterations reached — return partial answer
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
