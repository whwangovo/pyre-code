"""Runs one AI-track submission in isolation: load the learner's code, then each test case.

The grading service executes this file as ``python -I runner.py`` with a JSON job on stdin
and reads one JSON line per outcome from stdout as they are produced. Streaming the
outcomes lets the parent enforce a time limit per test and attribute a hang or a crash to
the test that caused it, instead of losing the whole submission. ``-I`` keeps the learner's
code from importing anything on sys.path other than the installed packages.

``run_job`` is also used in-process by the test-suite (scripts/test_tasks.py), where the
reference solutions are trusted and an interpreter start per task would dominate the run.

Job:
    {"code": str, "function_name": str, "capture": bool,
     "tests": [{"name": str, "code": str}]}
Outcomes, in order:
    {"load_error": str}     the code failed to load or lacks the function; no test ran
    {"loaded": true}        the code loaded; one outcome per test follows
    {"i": int, "name": str, "passed": bool, "execTimeMs": float,
     "error": str | None, "output": str | None}
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
from typing import Any, Iterator

MAX_OUTPUT_CHARS = 100_000


def _namespace() -> dict[str, Any]:
    """What the learner's code and the tests can use without importing."""
    import numpy as np
    import torch

    return {"torch": torch, "Tensor": torch.Tensor, "nn": torch.nn, "F": torch.nn.functional, "np": np, "math": math}


def _error_text(exc: BaseException) -> str:
    # Assertion messages are written for the learner and stand on their own
    if isinstance(exc, AssertionError):
        return str(exc) or "Assertion failed"
    return f"{type(exc).__name__}: {exc}"


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n… output truncated at {MAX_OUTPUT_CHARS} characters"


def run_job(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the outcomes of ``job`` in order; see the module docstring for their shapes."""
    fn_name = job["function_name"]
    capture = job.get("capture", True)
    real_stdout = sys.stdout

    # Load the learner's module. Prints at its top level belong to no test and are dropped.
    user_ns = _namespace()
    sys.stdout = io.StringIO()
    try:
        exec(compile(job["code"], "<solution>", "exec"), user_ns)
    except SyntaxError as exc:
        yield {"load_error": f"Syntax error: {exc}"}
        return
    except BaseException as exc:  # noqa: BLE001 - anything the learner's code raises is a load error
        yield {"load_error": _error_text(exc)}
        return
    finally:
        sys.stdout = real_stdout

    if fn_name not in user_ns:
        yield {"load_error": f"Function '{fn_name}' not found in submitted code"}
        return
    yield {"loaded": True}

    for i, test in enumerate(job["tests"]):
        test_ns = {**_namespace(), fn_name: user_ns[fn_name]}
        captured = io.StringIO()
        if capture:
            sys.stdout = captured
        start = time.perf_counter()
        try:
            exec(compile(test["code"], f"<test:{test['name']}>", "exec"), test_ns)
            passed, error = True, None
        except BaseException as exc:  # noqa: BLE001 - report whatever the test or the code raised
            passed, error = False, _error_text(exc)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            sys.stdout = real_stdout
        output = _clip(captured.getvalue()) if capture else ""
        yield {
            "i": i,
            "name": test["name"],
            "passed": passed,
            "execTimeMs": elapsed_ms,
            "error": error,
            "output": output or None,
        }


def main() -> None:
    job = json.loads(sys.stdin.read())
    # Written through the original stream: the learner's code may reassign sys.stdout
    out = sys.stdout
    for outcome in run_job(job):
        out.write(json.dumps(outcome) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
