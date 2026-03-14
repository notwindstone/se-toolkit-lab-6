"""Regression test for agent.py — Task 1.

Verifies that agent.py outputs valid JSON with required fields.
Does not call the real LLM API; checks structure only.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_agent_outputs_valid_json():
    """Run agent.py with a dummy question and verify JSON structure.

    This test assumes the environment is configured (.env.agent.secret exists).
    If the API call fails, we still check that the output is valid JSON
    with the required keys — the structure matters more than the content for Task 1.
    """
    # Run agent.py as subprocess using uv
    result = subprocess.run(
        [sys.executable, "-m", "uv", "run", "agent.py", "test question for structure check"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # project root
        timeout=65,  # 60s API timeout + buffer
    )

    # Parse stdout as JSON
    try:
        output = json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        pytest.fail(f"agent.py stdout is not valid JSON: {result.stdout[:200]}\nError: {e}")

    # Verify required fields exist with correct types
    assert "answer" in output, "Missing 'answer' field in output"
    assert isinstance(output["answer"], str), f"'answer' should be string, got {type(output['answer'])}"

    assert "tool_calls" in output, "Missing 'tool_calls' field in output"
    assert isinstance(output["tool_calls"], list), f"'tool_calls' should be list, got {type(output['tool_calls'])}"

    # For Task 1, tool_calls should be empty
    assert output["tool_calls"] == [], f"tool_calls should be empty array for Task 1, got {output['tool_calls']}"