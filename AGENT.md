# Task 1 Plan — Call an LLM from Code

**Date:** 2026-03-13  
**Task:** [Call an LLM from Code](../../lab/tasks/required/task-1.md)

---

## LLM Provider & Model

| Item | Value |
|------|-------|
| **Provider** | Qwen Code API (via OpenAI-compatible endpoint) |
| **Model** | `qwen3-coder-plus` |
| **Why** | 1000 free requests/day, no credit card required, works from Russia, strong tool-calling support |
| **API Endpoint** | `http://<vm-ip>:<port>/v1/chat/completions` |
| **Fallback** | OpenRouter (`meta-llama/llama-3.3-70b-instruct:free`) |

---

## Agent Structure

### Architecture Overview

CLI Input  ->  agent.py       ->  LLM API
(question)     (orchestrator)     (Qwen)


### Key Components

| Component | Description |
|-----------|-------------|
| **Input parsing** | Read question from `sys.argv[1]` |
| **Environment config** | Load `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL` from `.env.agent.secret` |
| **HTTP client** | Use `httpx` to call the LLM API with proper auth headers |
| **Response handling** | Parse JSON response, extract `answer` and `tool_calls` |
| **Output** | Print valid JSON to stdout, debug info to stderr |

### Data Flow

1. Read question from command line argument
2. Load API credentials from environment variables
3. Build chat completion request with minimal system prompt
4. Send request to LLM API with 60-second timeout
5. Parse response and extract required fields
6. Output JSON: `{"answer": "...", "tool_calls": []}`

### Error Handling

- Timeout after 60 seconds → exit code 1
- Invalid JSON response → exit code 1
- Missing required fields → exit code 1
- All errors logged to stderr (not stdout)
- Exit code 0 only on successful JSON output

---

## Environment Variables

| Variable | Purpose | Source |
|----------|---------|--------|
| `LLM_API_KEY` | LLM provider API key | `.env.agent.secret` |
| `LLM_API_BASE` | LLM API endpoint URL | `.env.agent.secret` |
| `LLM_MODEL` | Model name | `.env.agent.secret` |

**Note:** These are separate from `LMS_API_KEY` in `.env.docker.secret` (backend auth).

---

## Testing Strategy

| Test | Purpose |
|------|---------|
| `test_agent_outputs_valid_json` | Verify stdout is parseable JSON |
| `test_agent_has_required_fields` | Verify `answer` and `tool_calls` exist |
| `test_agent_handles_empty_question` | Verify graceful handling of edge cases |

Tests will run `agent.py` as a subprocess and validate the output structure.

---

## Deliverables Checklist

- [x] `plans/task-1.md` (this file) — committed before code
- [x] `agent.py` — CLI program in project root
- [x] `.env.agent.secret` — LLM credentials (gitignored)
- [x] `AGENT.md` — Architecture documentation
- [x] `test_agent.py` — 1 regression test
- [x] Git workflow: issue → branch → PR → review → merge