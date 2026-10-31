"""Tests for the PyTorch task set: every reference solution passes its own tests, and the
tests reject classic wrong solutions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from grading_service.main import _execute_tests
from torch_judge.tasks import TASKS


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_reference_solution_passes(task_id):
    task = TASKS[task_id]
    # In-process: the reference solutions are trusted, and a runner process per task would
    # turn a 3 s run into minutes. scripts/test_grading_service.py covers the isolation.
    result = _execute_tests(task["solution"], task, capture_output=False, isolated=False)
    failures = [(r.name, r.error) for r in result.results if not r.passed]
    assert result.allPassed, result.error or failures


# Classic wrong solutions: task_mutants/<task_id>/<mistake>.py, each opening with a
# "# Mistake: ..." comment; every one must fail at least one of the task's tests
MUTANTS_DIR = Path(__file__).parent / "task_mutants"
MIN_MUTANTS = 2
MUTANTS = sorted(MUTANTS_DIR.glob("*/*.py"))


@pytest.mark.parametrize("task_dir", sorted(p for p in MUTANTS_DIR.iterdir() if p.is_dir()), ids=lambda p: p.name)
def test_mutant_dir_names_a_task_and_has_enough_mutants(task_dir):
    assert task_dir.name in TASKS
    assert len(list(task_dir.glob("*.py"))) >= MIN_MUTANTS


@pytest.mark.parametrize("path", MUTANTS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_classic_mistake_is_caught(path):
    code = path.read_text(encoding="utf-8")
    assert code.startswith("# Mistake:")
    result = _execute_tests(code, TASKS[path.parent.name], capture_output=False, isolated=False)
    assert not result.allPassed, f"every test passes; first line: {code.splitlines()[0]}"
