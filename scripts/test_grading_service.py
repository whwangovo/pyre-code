"""Tests for the grading service's isolated AI-track runner: a test that hangs, crashes or
fails to load costs itself, not the submission or the service."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from grading_service import main as grading

TASK = {
    "function_name": "f",
    "tests": [
        {"name": "before", "code": "assert {fn}(1) == 1"},
        {"name": "bad", "code": "{fn}(None)"},
        {"name": "after", "code": "assert {fn}(2) == 2"},
    ],
}


@pytest.fixture(autouse=True)
def short_time_limit(monkeypatch):
    monkeypatch.setattr(grading, "TEST_LIMIT_S", 1.0)


def test_hanging_test_is_a_time_limit_not_a_hang():
    code = "def f(x):\n    while x is None:\n        pass\n    return x\n"
    result = grading._execute_tests(code, TASK, capture_output=False)
    assert [r.passed for r in result.results] == [True, False, True]
    assert result.results[1].error.startswith("Time Limit Exceeded")
    assert (result.passed, result.total, result.allPassed) == (2, 3, False)


def test_crash_in_one_test_leaves_the_others_graded():
    code = "import os\ndef f(x):\n    if x is None:\n        os._exit(3)\n    return x\n"
    result = grading._execute_tests(code, TASK, capture_output=False)
    assert [r.passed for r in result.results] == [True, False, True]
    assert result.results[1].error.startswith("Runtime Error")


def test_error_while_loading_is_reported_not_raised():
    result = grading._execute_tests("y = 1 / 0\ndef f(x):\n    return x\n", TASK, capture_output=False)
    assert result.results == [] and result.total == 0
    assert "ZeroDivisionError" in (result.error or "")


def test_missing_function_is_reported():
    result = grading._execute_tests("def g(x):\n    return x\n", TASK, capture_output=False)
    assert result.error == "Function 'f' not found in submitted code"


def test_print_output_is_captured_per_test():
    code = "def f(x):\n    print('hello', x)\n    return x\n"
    task = {"function_name": "f", "tests": [{"name": "t", "code": "assert {fn}(1) == 1"}]}
    result = grading._execute_tests(code, task)
    assert result.allPassed
    assert result.results[0].output == "hello 1\n"


def test_in_process_mode_matches_isolated_mode():
    code = "def f(x):\n    return x\n"
    isolated = grading._execute_tests(code, TASK, capture_output=False)
    in_process = grading._execute_tests(code, TASK, capture_output=False, isolated=False)
    assert [r.passed for r in isolated.results] == [r.passed for r in in_process.results] == [True, True, True]
