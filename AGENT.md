# Agent — Task 1: Call an LLM from Code

This document describes the `agent.py` CLI built for Task 1 of Lab 6.

## Overview

`agent.py` is a minimal CLI that:
1. Reads a question from the command line
2. Sends it to an LLM via an OpenAI-compatible API
3. Outputs a JSON response with `answer` and `tool_calls` fields

This is the foundation for the agentic loop you will build in Tasks 2–3.

## Architecture

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
