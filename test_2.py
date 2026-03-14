"""Regression tests for agent.py — Task 2: Documentation Agent.

Verifies that agent.py uses tools correctly and returns structured output.
Tests run agent.py as a subprocess and check JSON structure, tool usage, and source references.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_agent(question: str, timeout: int = 70) -> tuple[dict, str, str]:
    """Run agent.py and return (parsed_output, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "uv", "run", "agent.py", question],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # project root
        timeout=timeout,
    )
    try:
        output = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        pytest.fail(f"agent.py stdout is not valid JSON: {result.stdout[:200]}")
    return output, result.stdout, result.stderr


def test_wiki_lookup_merge_conflict():
    """Test that agent uses read_file to answer a wiki question about merge conflicts.
    
    Expected behavior:
    - Agent calls read_file tool to navigate wiki/
    - source field contains wiki/git-workflow.md reference
    - answer field contains relevant information about resolving conflicts
    """
    output, stdout, stderr = _run_agent("How do you resolve a merge conflict?")

    # Verify required fields exist with correct types
    assert "answer" in output, "Missing 'answer' field"
    assert isinstance(output["answer"], str), "'answer' must be string"
    assert len(output["answer"]) > 0, "'answer' must not be empty"

    assert "source" in output, "Missing 'source' field"
    assert isinstance(output["source"], str), "'source' must be string"

    assert "tool_calls" in output, "Missing 'tool_calls' field"
    assert isinstance(output["tool_calls"], list), "'tool_calls' must be list"

    # Verify tool usage: should have used read_file for wiki lookup
    tools_used = {tc.get("tool") for tc in output["tool_calls"]}
    assert "read_file" in tools_used, (
        f"Expected 'read_file' in tool_calls for wiki lookup, got: {tools_used}"
    )

    # Verify source points to git-workflow.md (may include section anchor)
    if output["source"]:
        assert "wiki/git-workflow.md" in output["source"], (
            f"Source should reference wiki/git-workflow.md, got: {output['source']}"
        )


def test_list_files_wiki():
    """Test that agent uses list_files to discover wiki contents.
    
    Expected behavior:
    - Agent calls list_files tool with path='wiki'
    - answer field contains information about available wiki files
    - tool_calls includes the list_files invocation with result
    """
    output, stdout, stderr = _run_agent("What files are in the wiki?")

    # Verify required fields exist with correct types
    assert "answer" in output, "Missing 'answer' field"
    assert isinstance(output["answer"], str), "'answer' must be string"

    assert "source" in output, "Missing 'source' field"
    assert isinstance(output["source"], str), "'source' must be string"

    assert "tool_calls" in output, "Missing 'tool_calls' field"
    assert isinstance(output["tool_calls"], list), "'tool_calls' must be list"

    # Verify tool usage: should have used list_files
    tools_used = {tc.get("tool") for tc in output["tool_calls"]}
    assert "list_files" in tools_used, (
        f"Expected 'list_files' in tool_calls for directory listing, got: {tools_used}"
    )

    # Verify list_files was called with wiki path
    list_files_found = False
    for tc in output["tool_calls"]:
        if tc.get("tool") == "list_files":
            list_files_found = True
            args = tc.get("args", {})
            assert args.get("path") == "wiki", (
                f"list_files should be called with path='wiki', got: {args}"
            )
            # Verify result was captured
            assert "result" in tc, "list_files tool call should include 'result' field"
            break
    
    assert list_files_found, "list_files tool call not found in tool_calls"