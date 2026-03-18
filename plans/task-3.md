# Task 3 Plan — The System Agent

**Date:** 2026-03-18  
**Task:** [The System Agent](../../lab/tasks/required/task-3.md)

---

## Tool Definitions

| Tool | Parameters | Return | Security |
|------|-----------|--------|----------|
| `query_api` | unsure | unsure | unsure |

### Tool Schema Format (OpenAI-compatible)

```json
{
  "type": "function",
  "function": {
    "name": "query_api",
    "description": "Query API.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "Answer the question using the known information and the system information"}
      },
      "required": ["path"]
    }
  }
}