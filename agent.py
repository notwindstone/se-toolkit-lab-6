#!/usr/bin/env python3
"""Agent CLI — Task 1: Call an LLM from code.

Outputs JSON with 'answer' and 'tool_calls' fields.
All debug output goes to stderr; only valid JSON to stdout.
"""
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv


def load_config() -> dict[str, str]:
    """Load LLM config from environment.

    Expects .env.agent.secret to be loaded via dotenv.
    Returns dict with LLM_API_KEY, LLM_API_BASE, LLM_MODEL.
    """
    # Load .env.agent.secret if it exists (project root)
    env_file = Path(__file__).parent / ".env.agent.secret"
    if env_file.exists():
        load_dotenv(env_file)

    required = ["LLM_API_KEY", "LLM_API_BASE", "LLM_MODEL"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        print("Copy .env.agent.example to .env.agent.secret and fill values.", file=sys.stderr)
        sys.exit(1)

    return {
        "api_key": os.environ["LLM_API_KEY"],
        "api_base": os.environ["LLM_API_BASE"].rstrip("/"),
        "model": os.environ["LLM_MODEL"],
    }


def call_llm(question: str, config: dict[str, str], timeout: int = 60) -> str:
    """Call the LLM API and return the assistant's response content.

    Args:
        question: The user's question.
        config: Dict with api_key, api_base, model.
        timeout: Request timeout in seconds.

    Returns:
        The text content from the LLM response.

    Raises:
        SystemExit on any error (with message to stderr).
    """
    url = f"{config['api_base']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": question}],
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        print(f"Request timed out after {timeout}s", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"HTTP error {e.response.status_code}: {e.response.text[:200]}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {response.text[:200]}", file=sys.stderr)
        sys.exit(1)

    # Extract answer from OpenAI-compatible response
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"Unexpected response structure: {data}", file=sys.stderr)
        sys.exit(1)

    return content


def main() -> None:
    """Entry point."""
    if len(sys.argv) < 2:
        print("Usage: uv run agent.py \"<question>\"", file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1]
    config = load_config()
    answer = call_llm(question, config)

    # Output valid JSON to stdout
    output = {
        "answer": answer,
        "tool_calls": [],  # Task 1: no tools yet
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
