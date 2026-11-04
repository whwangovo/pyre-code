"""Grade a Hot 100 submission in core or ACM mode.

Every case runs in its own ``python -I`` subprocess with a wall-clock limit, so an
infinite loop or a crash costs one case instead of the grading service.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from torch_judge.algo import runner, spec

TIME_LIMIT_S = 3.0
# Like core mode (and LeetCode), ACM programs may recurse over 10^4-deep trees and lists.
# The rlimits match the core runner's (see runner._apply_limits): a runaway program dies
# as its own per-case error instead of eating the host's memory.
ACM_BOOTSTRAP = (
    "import runpy, sys\n"
    "sys.setrecursionlimit(20000)\n"
    "try:\n"
    "    import resource\n"
    f"    resource.setrlimit(resource.RLIMIT_AS, ({runner.MEMORY_LIMIT_BYTES}, {runner.MEMORY_LIMIT_BYTES}))\n"
    f"    resource.setrlimit(resource.RLIMIT_CPU, ({runner.CPU_LIMIT_S}, {runner.CPU_LIMIT_HARD_S}))\n"
    "except ImportError:\n"
    "    pass\n"
    "runpy.run_path(sys.argv[1], run_name='__main__')"
)
PARALLEL_CASES = 4
RUNNER_PATH = Path(runner.__file__)
MAX_STDERR_LINES = 12


def _result(case: dict, task: dict, mode: str, **fields) -> dict:
    hidden = case.get("hidden", False)
    return {
        "name": case["name"],
        "input": None if hidden else spec.display_input(task, case, mode),
        "expected": None if hidden else spec.display_expected(task, case, mode),
        **fields,
    }


def _tle(case: dict, task: dict, mode: str) -> dict:
    return _result(case, task, mode, passed=False, execTimeMs=TIME_LIMIT_S * 1000,
                   error=f"Time Limit Exceeded (> {TIME_LIMIT_S:g}s)")


def _core_batch(code: str, task: dict, cases: list[dict]) -> list[dict] | None:
    """Run cases in one subprocess. None if it timed out or died, so the caller can rerun
    the cases one by one to pin the failure on a case."""
    job = {
        "code": code,
        "spec": spec.runner_spec(task),
        "cases": [{k: c[k] for k in ("args", "ops", "expected") if k in c} for c in cases],
    }
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(RUNNER_PATH)],
            input=json.dumps(job), capture_output=True, text=True, timeout=TIME_LIMIT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    wall_ms = (time.perf_counter() - start) * 1000
    try:
        outcomes = json.loads(proc.stdout.rstrip("\n").split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        if len(cases) > 1:
            return None
        stderr = _tail(proc.stderr) or "the program exited unexpectedly"
        return [_result(cases[0], task, "core", passed=False, execTimeMs=wall_ms, error=f"Runtime Error\n{stderr}")]

    results = []
    for case, outcome in zip(cases, outcomes):
        output = outcome.get("stdout") or None
        elapsed = outcome.get("timeMs", 0.0)
        if outcome["status"] == "error":
            results.append(_result(case, task, "core", passed=False, execTimeMs=elapsed, error=outcome["error"], output=output))
            continue
        passed = outcome["status"] == "ok"
        results.append(_result(case, task, "core", passed=passed, execTimeMs=elapsed, actual=outcome["actual"],
                               error=None if passed else "Wrong Answer", output=output))
    return results


def _run_core_case(code: str, task: dict, case: dict) -> dict:
    results = _core_batch(code, task, [case])
    return results[0] if results else _tle(case, task, "core")


def _run_core(code: str, task: dict, cases: list[dict]) -> list[dict]:
    # Small cases share one process (one interpreter start instead of ~25); the large
    # case gets its own time budget
    small = [c for c in cases if not c.get("hidden")]
    batch = _core_batch(code, task, small) if small else []
    if batch is None:
        with ThreadPoolExecutor(max_workers=PARALLEL_CASES) as pool:
            batch = list(pool.map(lambda c: _run_core_case(code, task, c), small))
    by_name = {r["name"]: r for r in batch}
    return [by_name[c["name"]] if not c.get("hidden") else _run_core_case(code, task, c) for c in cases]


def _tail(stderr: str, path: str | None = None) -> str:
    if path:
        stderr = stderr.replace(path, "solution.py")
    return "\n".join(stderr.strip().split("\n")[-MAX_STDERR_LINES:])


def _run_acm_case(program: Path, task: dict, case: dict) -> dict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", ACM_BOOTSTRAP, str(program)],
            input=spec.acm_stdin(task, case), capture_output=True, text=True,
            timeout=TIME_LIMIT_S, cwd=program.parent,
        )
    except subprocess.TimeoutExpired:
        return _tle(case, task, "acm")
    elapsed = (time.perf_counter() - start) * 1000
    actual = runner._clip(proc.stdout.rstrip("\n"))

    if proc.returncode != 0:
        return _result(case, task, "acm", passed=False, execTimeMs=elapsed, actual=actual or None,
                       error=f"Runtime Error\n{_tail(proc.stderr, str(program))}")

    try:
        passed = _acm_matches(task, case, proc.stdout)
        error = None if passed else "Wrong Answer"
    except (ValueError, NotImplementedError) as exc:
        passed, error = False, f"Wrong Answer (output format: {exc})"
    return _result(case, task, "acm", passed=passed, execTimeMs=elapsed, actual=actual, error=error)


def _acm_matches(task: dict, case: dict, stdout: str) -> bool:
    if task.get("design"):
        # One line per non-void call, parsed by that method's return type (floats compare
        # numerically, so "2.5" and "2.50000" both match)
        got = [line.strip() for line in stdout.strip().split("\n")] if stdout.strip() else []
        want = spec.design_outputs(task, case)
        if len(got) != len(want):
            return False
        return all(runner.check("approx", runner.parse_acm_output(ret, line), value)
                   for line, (ret, value) in zip(got, want))
    out_type = spec.acm_output_type(task)
    task_spec = spec.runner_spec(task)
    parsed = runner.parse_acm_output(out_type, stdout)
    if runner.check_answer(task_spec, case["args"], parsed, case["expected"]):
        return True
    # A nested answer may end with an empty row ([] printed last), which reads as a
    # trailing blank line; try again keeping it
    if out_type in ("int[][]", "str[][]"):
        parsed = runner.parse_acm_output(out_type, stdout, keep_empty_rows=True)
        return runner.check_answer(task_spec, case["args"], parsed, case["expected"])
    return False


def grade(task_id: str, task: dict, code: str, mode: str = "core", test_indices: list[int] | None = None) -> dict:
    if mode not in spec.MODES:
        return {"passed": 0, "total": 0, "allPassed": False, "results": [], "totalTimeMs": 0.0, "error": f"Unknown mode '{mode}'"}

    cases = spec.cases_for_mode(task_id, task, mode)
    if test_indices is not None:
        cases = [cases[i] for i in test_indices if 0 <= i < len(cases)]

    if mode == "acm":
        # Each ACM case needs its own process (its own stdin); run a few at once
        with tempfile.TemporaryDirectory(prefix="pyre-acm-") as tmp:
            program = Path(tmp) / "solution.py"
            program.write_text(code, encoding="utf-8")
            with ThreadPoolExecutor(max_workers=PARALLEL_CASES) as pool:
                results = list(pool.map(lambda c: _run_acm_case(program, task, c), cases))
    else:
        results = _run_core(code, task, cases)

    passed = sum(r["passed"] for r in results)
    return {
        "passed": passed,
        "total": len(results),
        "allPassed": passed == len(results),
        "results": results,
        "totalTimeMs": sum(r["execTimeMs"] for r in results),
    }
