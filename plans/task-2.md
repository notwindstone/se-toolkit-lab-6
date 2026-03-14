# Task 2 Plan — The Documentation Agent

**Date:** 2026-03-14  
**Task:** [The Documentation Agent](../../lab/tasks/required/task-2.md)

---

## Tool Definitions

| Tool | Parameters | Return | Security |
|------|-----------|--------|----------|
| `read_file` | `path: str` (relative to project root) | `str` (file contents) or error message | Block `..` traversal, resolve to absolute path, verify within project root |
| `list_files` | `path: str` (relative directory) | `str` (newline-separated entries) | Same path validation as `read_file` |

### Tool Schema Format (OpenAI-compatible)

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file from the project repository. Use to find answers in documentation or source code.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "Relative path from project root, e.g., 'wiki/git-workflow.md'"}
      },
      "required": ["path"]
    }
  }
}