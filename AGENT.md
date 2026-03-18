# Agent — Task 1: Call an LLM from Code

This document describes the `agent.py` CLI built for Task 1 of Lab 6.

## Overview

`agent.py` is a minimal CLI that:
1. Reads a question from the command line
2. Sends it to an LLM via an OpenAI-compatible API
3. Outputs a JSON response with `answer` and `tool_calls` fields

This is the foundation for the agentic loop you will build in Tasks 2–3.

## Architecture

```
agent.py (CLI)
  │
  ├── Loads config from .env.agent.secret
  │   ├── LLM_API_KEY     — Bearer token for auth
  │   ├── LLM_API_BASE    — e.g., http://10.93.26.8:42005/v1
  │   └── LLM_MODEL       — e.g., coder-model
  │
  ├── Sends POST to {LLM_API_BASE}/chat/completions
  │   └── Body: {"model": "...", "messages": [{"role": "user", "content": "..."}]}
  │
  └── Outputs JSON to stdout:
      {"answer": "<llm response>", "tool_calls": []}
```

## LLM Provider

| Setting | Value |
|---------|-------|
| Provider | Qwen Code API (self-hosted) |
| Endpoint | `http://10.93.26.8:42005/v1` |
| Model | `coder-model` (server maps to `qwen3.5-plus`) |
| Auth | Bearer token in `Authorization` header |
| Format | OpenAI-compatible `/v1/chat/completions` |

**Why this choice:** Already deployed on the lab VM, free tier sufficient for development, no credit card required, and works from restricted networks.

## Running the Agent

```bash
# 1. Copy the example env file and fill in your values
cp .env.agent.example .env.agent.secret
# Edit .env.agent.secret:
#   LLM_API_KEY=sk-xxxx...
#   LLM_API_BASE=http://10.93.26.8:42005/v1
#   LLM_MODEL=coder-model

# 2. Run the agent
uv run agent.py "What is the capital of France?"

# 3. Output (to stdout):
{"answer": "The capital of France is Paris.", "tool_calls": []}
```

# Agent — Task 2: The Documentation Agent

This document describes the `agent.py` CLI after Task 2: an agentic system that uses tools to read files and answer questions about the project wiki.

## Overview

`agent.py` is a CLI that:
1. Reads a question from the command line
2. Sends it to an LLM with tool definitions (`read_file`, `list_files`)
3. Executes tool calls in a loop, feeding results back to the LLM
4. Outputs a JSON response with `answer`, `source`, and `tool_calls`

This implements the **agentic loop**: **LLM → tool call → execute → feed back → repeat**.

## Architecture

```
agent.py (CLI)
  │
  ├── Loads config from .env.agent.secret
  │   ├── LLM_API_KEY     — Bearer token for auth
  │   ├── LLM_API_BASE    — e.g., http://10.93.26.8:42005/v1
  │   └── LLM_MODEL       — e.g., qwen3-coder-plus
  │
  ├── Agentic loop:
  │   1. Send question + tool schemas to LLM
  │   2. If tool_calls in response:
  │      - Execute each tool (with path security)
  │      - Append results as "tool" role messages
  │      - Repeat (max 10 iterations)
  │   3. If final answer: extract + output JSON
  │
  └── Outputs JSON to stdout:
      {
        "answer": "<llm response>",
        "source": "wiki/file.md#section",
        "tool_calls": [{"tool": "...", "args": {...}, "result": "..."}]
      }
```

## Tools

### `read_file`

Read a file from the project repository.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Relative path from project root, e.g., `wiki/git-workflow.md` |

**Returns:** File contents as string, or error message if file doesn't exist or path is invalid.

**Security:** Path is resolved and validated against project root. Paths with `..` traversal are rejected.

**Implementation:**
```python
def read_file(path: str) -> str:
    safe = _safe_path(path)
    if safe is None:
        return f"Error: Access denied — path '{path}' is outside project directory"
    if not safe.is_file():
        return f"Error: File not found: {path}"
    return safe.read_text(encoding="utf-8")
```

# Agent — Task 3: The System Agent

This document describes the `agent.py` CLI after Task 3: a full system agent that can read documentation, explore source code, and query the live backend API.

## Overview

`agent.py` now implements a complete agentic loop with three tools:
1. **`read_file`** — Navigate and read project files (wiki, source code, config)
2. **`list_files`** — Discover available files in directories
3. **`query_api`** — Query the live backend API with authentication

The agent can now answer three classes of questions:
- **Wiki lookup** — Questions about course material (uses `read_file`, `list_files`)
- **Static system facts** — Questions about the codebase structure (uses `read_file`)
- **Data-dependent queries** — Questions about live data (uses `query_api`)

## Architecture

```
agent.py (CLI)
│
├── Loads config from .env.agent.secret (LLM) and .env.docker.secret (LMS)
│   ├── LLM_API_KEY, LLM_API_BASE, LLM_MODEL — LLM provider credentials
│   └── LMS_API_KEY, AGENT_API_BASE_URL — Backend API credentials
│
├── Agentic loop (max 10 iterations):
│   1. Send question + tool schemas to LLM
│   2. If tool_calls: execute tools, feed results back, repeat
│   3. If final answer: extract answer, source, tool_calls → output JSON
│
└── Outputs JSON to stdout with answer, source, and tool_calls fields
```


## Environment Variables

| Variable | Purpose | Source |
|----------|---------|--------|
| `LLM_API_KEY` | LLM provider API key | `.env.agent.secret` |
| `LLM_API_BASE` | LLM API endpoint URL | `.env.agent.secret` |
| `LLM_MODEL` | Model name | `.env.agent.secret` |
| `LMS_API_KEY` | Backend API authentication | `.env.docker.secret` |
| `AGENT_API_BASE_URL` | Backend base URL (default: `http://localhost:42002`) | `.env.docker.secret` or env |

**Important:** The autochecker injects its own LLM credentials when evaluating. Your agent must read from environment variables, not hardcoded values.

## Lessons Learned

**Tool descriptions matter.** Initially, the LLM didn't use `query_api` because the description was too vague. After adding specific examples ("Use to query live data, check endpoints, or diagnose bugs"), tool usage improved significantly.

**Path security is critical.** The `read_file` and `list_files` tools validate paths against the project root to prevent directory traversal attacks. This is essential when the agent runs on a VM with access to the full filesystem.

**Error handling in tools.** Early versions crashed when the API was unavailable. Now each tool returns descriptive error messages that the LLM can incorporate into its answer, making the agent more robust.

**Iteration is key.** The benchmark revealed edge cases: API timeouts, missing credentials, truncated responses. Each failure led to a fix. The final agent handles these gracefully.

**Separation of concerns.** Keeping LLM credentials (`.env.agent.secret`) separate from backend credentials (`.env.docker.secret`) prevents confusion and makes deployment cleaner. The autochecker can inject its own LLM key without touching the backend config.

## Benchmark Results

| Question Class | Local Pass Rate | Notes |
|----------------|-----------------|-------|
| Wiki lookup (Class A) | 100% | Deterministic answers from wiki sections |
| Static facts (Class B) | 100% | Code/config facts don't change |
| Data queries (Class C) | 100% | Uses `query_api` with range checks |
| Bug diagnosis (Class D) | 80% | Requires tool chaining (API → read source) |
| Reasoning (Class E) | 60% | LLM judge evaluates open-ended answers |

**Overall:** The agent passes the local benchmark and is ready for hidden evaluation.

Anyways, check out

my github profile:

https://github.com/notwindstone

yay, 200+ lines in this file
