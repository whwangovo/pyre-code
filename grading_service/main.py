"""FastAPI grading service for torch_judge tasks."""

import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path for torch_judge imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from torch_judge.tasks import get_task

app = FastAPI(title="Grading Service")

# ---------------------------------------------------------------------------
# SQLite DB (user sessions + progress)
# ---------------------------------------------------------------------------

_DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).parent.parent / "data" / "pyre.db"))


def _get_db() -> sqlite3.Connection:
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_token TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('todo', 'attempted', 'solved')),
            best_time_ms REAL,
            attempts INTEGER DEFAULT 0,
            solved_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, task_id)
        )
    """)
    conn.commit()
    return conn


class SubmitRequest(BaseModel):
    taskId: str
    code: str


class RunRequest(BaseModel):
    taskId: str
    code: str
    testIndices: list[int] | None = None


class TestResult(BaseModel):
    name: str
    passed: bool
    execTimeMs: float
    error: str | None = None
    output: str | None = None


class GradeResponse(BaseModel):
    passed: int
    total: int
    allPassed: bool
    results: list[TestResult]
    totalTimeMs: float
    error: str | None = None


def _validate_code(code: str) -> str | None:
    """Return an error message if code contains disallowed top-level statements."""
    import ast
    allowed = (
        ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import, ast.ImportFrom,
        ast.Assign, ast.AnnAssign, ast.AugAssign,
        ast.Expr,  # top-level expressions / docstrings
    )
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"
    for node in tree.body:
        if not isinstance(node, allowed):
            return f"Only definitions and assignments are allowed at the top level (found: {type(node).__name__})"
    return None


def _execute_tests(code: str, task: dict, test_indices: list[int] | None = None, capture_output: bool = True) -> GradeResponse:
    import torch, math
    err = _validate_code(code)
    if err:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error=err)
    user_ns: dict[str, Any] = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "np": __import__("numpy"),
        "math": math,
    }
    try:
        exec(code, user_ns)
    except SyntaxError as e:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error=f"Syntax error: {e}")

    fn_name = task.get("function_name")
    if fn_name is None:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error="Task has no function_name defined")

    if fn_name not in user_ns:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error=f"Function '{fn_name}' not found in submitted code")

    all_tests = task.get("tests", [])
    tests = [all_tests[i] for i in test_indices if i < len(all_tests)] if test_indices is not None else all_tests

    results: list[TestResult] = []
    passed = 0
    total_time_ms = 0.0

    for test in tests:
        _torch = __import__("torch")
        test_ns: dict[str, Any] = {
            "torch": _torch,
            "Tensor": _torch.Tensor,
            "nn": _torch.nn,
            "F": _torch.nn.functional,
            "np": __import__("numpy"),
            "math": math,
            fn_name: user_ns[fn_name],
        }
        test_code = test["code"].replace("{fn}", fn_name)

        # Capture stdout for print output
        output = None
        if capture_output:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured = io.StringIO()
            try:
                start = time.perf_counter()
                exec(test_code, test_ns)
                exec_time_ms = (time.perf_counter() - start) * 1000
                output = captured.getvalue() or None
                results.append(TestResult(name=test["name"], passed=True, execTimeMs=exec_time_ms, output=output))
                passed += 1
            except AssertionError as e:
                exec_time_ms = (time.perf_counter() - start) * 1000
                output = captured.getvalue() or None
                results.append(TestResult(name=test["name"], passed=False, execTimeMs=exec_time_ms, error=str(e), output=output))
            except Exception as e:
                exec_time_ms = (time.perf_counter() - start) * 1000
                output = captured.getvalue() or None
                results.append(TestResult(name=test["name"], passed=False, execTimeMs=exec_time_ms, error=f"{type(e).__name__}: {e}", output=output))
            finally:
                sys.stdout = old_stdout
        else:
            start = time.perf_counter()
            try:
                exec(test_code, test_ns)
                exec_time_ms = (time.perf_counter() - start) * 1000
                results.append(TestResult(name=test["name"], passed=True, execTimeMs=exec_time_ms))
                passed += 1
            except AssertionError as e:
                exec_time_ms = (time.perf_counter() - start) * 1000
                results.append(TestResult(name=test["name"], passed=False, execTimeMs=exec_time_ms, error=str(e)))
            except Exception as e:
                exec_time_ms = (time.perf_counter() - start) * 1000
                results.append(TestResult(name=test["name"], passed=False, execTimeMs=exec_time_ms, error=f"{type(e).__name__}: {e}"))
        total_time_ms += exec_time_ms

    return GradeResponse(passed=passed, total=len(results), allPassed=passed == len(results), results=results, totalTimeMs=total_time_ms)


@app.post("/grade", response_model=GradeResponse)
def grade(request: SubmitRequest) -> GradeResponse:
    task = get_task(request.taskId)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{request.taskId}' not found")
    return _execute_tests(request.code, task)


@app.post("/run", response_model=GradeResponse)
def run(request: RunRequest) -> GradeResponse:
    task = get_task(request.taskId)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{request.taskId}' not found")
    return _execute_tests(request.code, task, request.testIndices)


def _classify_cell(src: str, fn_name: str | None) -> str:
    """Classify a notebook code cell as 'solution' or 'demo' by actually executing it."""
    import torch, math
    if not fn_name:
        return "demo"
    ns: dict[str, Any] = {
        "torch": torch, "Tensor": torch.Tensor,
        "nn": torch.nn, "F": torch.nn.functional,
        "np": __import__("numpy"), "math": math,
    }
    try:
        exec(src, ns)
    except Exception:
        pass
    return "solution" if fn_name in ns and callable(ns[fn_name]) else "demo"


@app.get("/tasks/{task_id}/notebook")
def get_notebook(task_id: str) -> dict:
    import json as _json
    task = get_task(task_id)
    fn_name = task.get("function_name") if task else None
    solutions_dir = Path(__file__).parent.parent / "solutions"
    matches = list(solutions_dir.glob(f"*_{task_id}_solution.ipynb"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Notebook for '{task_id}' not found")
    nb = _json.loads(matches[0].read_text())
    _skip = ("google.colab", "torch_judge", "get_ipython", "colab.research.google.com")
    def _strip_imports(src: str) -> str:
        lines = [l for l in src.splitlines() if not l.startswith("import ") and not l.startswith("from ")]
        return "\n".join(lines).strip()

    def _strip_comment_lines(src: str) -> str:
        lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
        return "\n".join(lines).strip()
    cells = []
    for c in nb.get("cells", []):
        src = "".join(c["source"])
        if not src.strip() or any(s in src for s in _skip):
            continue
        if c["cell_type"] == "code":
            src = _strip_imports(_strip_comment_lines(src))
            if not src.strip():
                continue
            role = _classify_cell(src, fn_name)
        else:
            role = "explanation"
        cells.append({"type": c["cell_type"], "source": src, "role": role})
    return {"cells": cells}


@app.get("/tasks/{task_id}/solution")
def get_solution(task_id: str) -> dict[str, str]:
    task = get_task(task_id)
    if task is None or not task.get("solution"):
        raise HTTPException(status_code=404, detail=f"Solution for '{task_id}' not found")
    return {"solution": task["solution"]}


class UserRequest(BaseModel):
    sessionToken: str


class ProgressEntry(BaseModel):
    status: str
    bestTimeMs: float | None = None
    attempts: int
    solvedAt: str | None = None


class SaveProgressRequest(BaseModel):
    sessionToken: str
    taskId: str
    status: str
    execTimeMs: float | None = None


@app.post("/users")
def get_or_create_user(request: UserRequest) -> dict[str, int]:
    with _get_db() as conn:
        row = conn.execute("SELECT id FROM users WHERE session_token = ?", (request.sessionToken,)).fetchone()
        if row:
            return {"userId": row[0]}
        cur = conn.execute("INSERT INTO users (session_token) VALUES (?)", (request.sessionToken,))
        return {"userId": cur.lastrowid}


@app.get("/progress/{user_id}")
def get_progress(user_id: int) -> dict[str, ProgressEntry]:
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT task_id, status, best_time_ms, attempts, solved_at FROM progress WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    return {
        row[0]: ProgressEntry(status=row[1], bestTimeMs=row[2], attempts=row[3], solvedAt=row[4])
        for row in rows
    }


@app.post("/progress")
def save_progress(request: SaveProgressRequest) -> dict[str, str]:
    with _get_db() as conn:
        row = conn.execute("SELECT id FROM users WHERE session_token = ?", (request.sessionToken,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        user_id = row[0]
        existing = conn.execute(
            "SELECT status, best_time_ms FROM progress WHERE user_id = ? AND task_id = ?",
            (user_id, request.taskId)
        ).fetchone()
        if existing:
            if request.status == "solved":
                best = min(existing[1], request.execTimeMs) if existing[1] and request.execTimeMs else (existing[1] or request.execTimeMs)
                conn.execute(
                    "UPDATE progress SET status = ?, best_time_ms = ?, attempts = attempts + 1, solved_at = datetime('now') WHERE user_id = ? AND task_id = ?",
                    (request.status, best, user_id, request.taskId)
                )
            else:
                conn.execute(
                    "UPDATE progress SET status = ?, attempts = attempts + 1 WHERE user_id = ? AND task_id = ?",
                    (request.status, user_id, request.taskId)
                )
        else:
            conn.execute(
                "INSERT INTO progress (user_id, task_id, status, best_time_ms, attempts, solved_at) VALUES (?, ?, ?, ?, 1, datetime('now'))",
                (user_id, request.taskId, request.status, request.execTimeMs)
            )
    return {"ok": "true"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


