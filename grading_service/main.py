"""FastAPI grading service for torch_judge tasks."""

import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path for torch_judge imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import queue
import signal
import subprocess
import threading
from collections import deque
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from grading_service import runner
from torch_judge.algo import judge as algo_judge
from torch_judge.tasks import get_task

# ---------------------------------------------------------------------------
# SQLite DB (user sessions + progress)
# ---------------------------------------------------------------------------

_DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).parent.parent / "data" / "pyre.db"))


def _init_db() -> None:
    """Create the schema and run migrations once, at startup (see lifespan)."""
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                code TEXT NOT NULL,
                passed INTEGER NOT NULL,
                exec_time_ms REAL,
                submitted_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        # Additive migrations; guarded because `uvicorn --workers 2` runs this twice
        columns = {row[1] for row in conn.execute("PRAGMA table_info(submissions)")}
        for name, ddl in (
            # Answer mode ('core' or 'acm') of Hot 100 submissions; older rows default to core
            ("mode", "ALTER TABLE submissions ADD COLUMN mode TEXT NOT NULL DEFAULT 'core'"),
            # Free-text note the learner attaches to a submission
            ("note", "ALTER TABLE submissions ADD COLUMN note TEXT NOT NULL DEFAULT ''"),
        ):
            if name not in columns:
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # the other worker won the race
        # Unsubmitted editor code, so a draft follows the learner across devices.
        # updated_at is the client's edit time in epoch ms; the newest edit wins.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drafts (
                user_id INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                code TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                PRIMARY KEY (user_id, task_id, mode)
            )
        """)
        # The AI help conversation for each problem, as a JSON array of chat messages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_chats (
                user_id INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                messages TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id),
                PRIMARY KEY (user_id, task_id)
            )
        """)
        # History reads filter by (user_id, task_id)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_submissions_user_task ON submissions(user_id, task_id)"
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _init_db()
    yield


app = FastAPI(title="Grading Service", lifespan=lifespan)


def _get_db() -> sqlite3.Connection:
    """A fresh connection with its pragmas; the schema is set up once in lifespan."""
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _user_id_for_token(conn: sqlite3.Connection, session_token: str) -> int | None:
    row = conn.execute("SELECT id FROM users WHERE session_token = ?", (session_token,)).fetchone()
    return row[0] if row else None


def _get_or_create_user(conn: sqlite3.Connection, session_token: str) -> int:
    """Write paths may be a learner's first action; the token then mints the user."""
    user_id = _user_id_for_token(conn, session_token)
    if user_id is not None:
        return user_id
    cur = conn.execute("INSERT INTO users (session_token) VALUES (?)", (session_token,))
    return cur.lastrowid


class SubmitRequest(BaseModel):
    taskId: str
    code: str
    mode: str = "core"


class RunRequest(BaseModel):
    taskId: str
    code: str
    testIndices: list[int] | None = None
    mode: str = "core"


class TestResult(BaseModel):
    name: str
    passed: bool
    execTimeMs: float
    error: str | None = None
    output: str | None = None
    # Hot 100 cases: display strings for the case and the learner's answer
    input: str | None = None
    expected: str | None = None
    actual: str | None = None


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


# ---------------------------------------------------------------------------
# AI-track execution: an isolated runner process per submission (see runner.py)
# ---------------------------------------------------------------------------

RUNNER_PATH = Path(runner.__file__)
# Interpreter start, importing torch and loading the learner's code, before the first test
LOAD_BUDGET_S = float(os.environ.get("AI_JUDGE_LOAD_BUDGET_S", "30"))
# Wall-clock limit for one test case
TEST_LIMIT_S = float(os.environ.get("AI_JUDGE_TEST_LIMIT_S", "10"))
MAX_STDERR_LINES = 12


def _spawn_runner(job: dict[str, Any]) -> subprocess.Popen:
    # Its own session (process group), so a kill also reaches anything the learner's code spawned
    proc = subprocess.Popen(
        [sys.executable, "-I", str(RUNNER_PATH)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", start_new_session=(os.name == "posix"),
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(job))
        proc.stdin.close()
    except OSError:
        pass  # it died before reading; the missing outcomes are reported by the caller
    return proc


def _kill_runner(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _pump_lines(stream, out: queue.Queue) -> None:
    try:
        for line in stream:
            out.put(line)
    except (OSError, ValueError):
        pass
    finally:
        out.put(None)  # EOF: the process ended


def _pump_stderr(stream, tail: deque) -> None:
    # Drained in its own thread so a chatty process can't block on a full stderr pipe
    try:
        for line in stream:
            tail.append(line.rstrip("\n"))
    except (OSError, ValueError):
        pass


def _failed(test: dict[str, Any], error: str) -> dict[str, Any]:
    return {"name": test["name"], "passed": False, "execTimeMs": 0.0, "error": error, "output": None}


def _run_job_isolated(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Run ``job`` in runner subprocesses and return its outcomes (shapes in runner.py).

    One process normally runs every test. When a test exceeds TEST_LIMIT_S or takes the
    process down with it, that test is recorded as failed and the remaining tests continue
    in a fresh process: one bad test costs itself, not the submission or the service.
    """
    tests = job["tests"]
    outcomes: list[dict[str, Any]] = []
    while len(outcomes) < len(tests):
        first = len(outcomes)
        proc = _spawn_runner({**job, "tests": tests[first:]})
        lines: queue.Queue = queue.Queue()
        stderr_tail: deque = deque(maxlen=MAX_STDERR_LINES)
        pumps = [
            threading.Thread(target=_pump_lines, args=(proc.stdout, lines), daemon=True),
            threading.Thread(target=_pump_stderr, args=(proc.stderr, stderr_tail), daemon=True),
        ]
        for pump in pumps:
            pump.start()
        loaded = False
        try:
            while len(outcomes) < len(tests):
                current = tests[len(outcomes)]
                try:
                    line = lines.get(timeout=TEST_LIMIT_S if loaded else LOAD_BUDGET_S)
                except queue.Empty:
                    if not loaded and first == 0:
                        return [{"load_error": f"Time Limit Exceeded: the code did not finish loading within {LOAD_BUDGET_S:g}s"}]
                    outcomes.append(_failed(current, f"Time Limit Exceeded (> {TEST_LIMIT_S:g}s)"))
                    break
                if line is None:
                    detail = "\n".join(stderr_tail) or "the process exited unexpectedly"
                    if not loaded and first == 0:
                        return [{"load_error": f"Runtime Error\n{detail}"}]
                    outcomes.append(_failed(current, f"Runtime Error\n{detail}"))
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # something wrote to the real stdout; the outcome line follows
                if item.get("loaded"):
                    loaded = True
                elif "load_error" in item:
                    if first == 0:
                        return [item]
                    outcomes.append(_failed(current, item["load_error"]))
                    break
                else:
                    outcomes.append({**item, "name": current["name"]})
        finally:
            _kill_runner(proc)
            for pump in pumps:
                pump.join(timeout=2)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
    return outcomes


def _execute_tests(
    code: str,
    task: dict,
    test_indices: list[int] | None = None,
    capture_output: bool = True,
    isolated: bool = True,
) -> GradeResponse:
    """Grade ``code`` against an AI-track task.

    ``isolated=False`` runs the tests in this process. It exists for the test-suite, where the
    reference solutions are trusted and an interpreter per task would dominate the run; the
    service always isolates.
    """
    err = _validate_code(code)
    if err:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error=err)

    fn_name = task.get("function_name")
    if fn_name is None:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error="Task has no function_name defined")

    all_tests = task.get("tests", [])
    tests = [all_tests[i] for i in test_indices if 0 <= i < len(all_tests)] if test_indices is not None else all_tests

    if test_indices is not None and len(test_indices) > 0 and len(tests) == 0:
        return GradeResponse(
            passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0,
            error=f"All provided test indices are out of range (valid range: 0..{len(all_tests) - 1})",
        )

    job = {
        "code": code,
        "function_name": fn_name,
        "capture": capture_output,
        "tests": [{"name": test["name"], "code": test["code"].replace("{fn}", fn_name)} for test in tests],
    }
    outcomes = _run_job_isolated(job) if isolated else list(runner.run_job(job))
    if outcomes and "load_error" in outcomes[0]:
        return GradeResponse(passed=0, total=0, allPassed=False, results=[], totalTimeMs=0.0, error=outcomes[0]["load_error"])

    results = [
        TestResult(name=o["name"], passed=o["passed"], execTimeMs=o["execTimeMs"], error=o.get("error"), output=o.get("output"))
        for o in outcomes if "name" in o
    ]
    passed = sum(1 for r in results if r.passed)
    return GradeResponse(
        passed=passed, total=len(results), allPassed=passed == len(results), results=results,
        totalTimeMs=sum(r.execTimeMs for r in results),
    )


def _grade(task_id: str, code: str, mode: str, test_indices: list[int] | None = None) -> GradeResponse:
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if task.get("track") == "hot100":
        return GradeResponse(**algo_judge.grade(task_id, task, code, mode, test_indices))
    return _execute_tests(code, task, test_indices)


@app.post("/grade", response_model=GradeResponse)
def grade(request: SubmitRequest) -> GradeResponse:
    return _grade(request.taskId, request.code, request.mode)


@app.post("/run", response_model=GradeResponse)
def run(request: RunRequest) -> GradeResponse:
    return _grade(request.taskId, request.code, request.mode, request.testIndices)



@app.get("/tasks/{task_id}/notebook")
def get_notebook(task_id: str) -> dict:
    task = get_task(task_id)
    if task is None or not task.get("solution"):
        raise HTTPException(status_code=404, detail=f"Notebook for '{task_id}' not found")
    cells = [{"type": "code", "source": task["solution"].strip(), "role": "solution"}]
    if "explanation" in task:
        cells.append({"type": "markdown", "source": task["explanation"].strip(), "role": "explanation"})
    if "demo" in task:
        cells.append({"type": "code", "source": task["demo"].strip(), "role": "demo"})
    return {"cells": cells}


@app.get("/tasks/{task_id}/solution")
def get_solution(task_id: str) -> dict[str, str]:
    task = get_task(task_id)
    if task is None or not task.get("solution"):
        raise HTTPException(status_code=404, detail=f"Solution for '{task_id}' not found")
    return {"solution": task["solution"]}


class ProgressEntry(BaseModel):
    status: str
    bestTimeMs: float | None = None
    attempts: int
    solvedAt: str | None = None


class SubmitAndSaveRequest(BaseModel):
    sessionToken: str
    taskId: str
    code: str
    mode: str = "core"


def _record_submission(
    conn: sqlite3.Connection,
    user_id: int,
    task_id: str,
    status: str,
    exec_time_ms: float | None,
    code: str,
    all_passed: bool,
    mode: str,
) -> None:
    """Update progress and append to the submission history; the caller owns the transaction."""
    existing = conn.execute(
        "SELECT status, best_time_ms FROM progress WHERE user_id = ? AND task_id = ?",
        (user_id, task_id),
    ).fetchone()
    if existing:
        existing_status, existing_best_time = existing
        if status == "solved":
            if existing_best_time is not None and exec_time_ms is not None:
                best = min(existing_best_time, exec_time_ms)
            else:
                best = existing_best_time if existing_best_time is not None else exec_time_ms
            conn.execute(
                "UPDATE progress SET status = ?, best_time_ms = ?, attempts = attempts + 1, solved_at = COALESCE(solved_at, datetime('now')) WHERE user_id = ? AND task_id = ?",
                ("solved", best, user_id, task_id),
            )
        else:
            next_status = existing_status if existing_status == "solved" and status in ("todo", "attempted") else status
            conn.execute(
                "UPDATE progress SET status = ?, attempts = attempts + 1 WHERE user_id = ? AND task_id = ?",
                (next_status, user_id, task_id),
            )
    elif status == "solved":
        conn.execute(
            "INSERT INTO progress (user_id, task_id, status, best_time_ms, attempts, solved_at) VALUES (?, ?, ?, ?, 1, datetime('now'))",
            (user_id, task_id, status, exec_time_ms),
        )
    else:
        conn.execute(
            "INSERT INTO progress (user_id, task_id, status, best_time_ms, attempts, solved_at) VALUES (?, ?, ?, ?, 1, NULL)",
            (user_id, task_id, status, None),
        )
    conn.execute(
        "INSERT INTO submissions (user_id, task_id, code, passed, exec_time_ms, mode) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, task_id, code, 1 if all_passed else 0, exec_time_ms, mode),
    )


@app.post("/submit", response_model=GradeResponse)
def submit_and_save(request: SubmitAndSaveRequest) -> GradeResponse:
    """Grade a submission and persist everything it implies in one transaction.

    Replaces the web layer's old /grade + /users + /progress chain, which could
    grade a submission and then fail to save it, or the reverse.
    """
    result = _grade(request.taskId, request.code, request.mode)
    status = "solved" if result.allPassed else "attempted"
    with _get_db() as conn:
        user_id = _get_or_create_user(conn, request.sessionToken)
        _record_submission(
            conn, user_id, request.taskId, status, result.totalTimeMs,
            request.code, result.allPassed, request.mode,
        )
    return result


@app.get("/progress")
def get_progress(session_token: str | None = Header(default=None, alias="X-Session-Token")) -> dict[str, ProgressEntry]:
    """The learner's progress by task id. A GET never creates a user: an unknown
    or missing token reads as empty."""
    if not session_token:
        return {}
    with _get_db() as conn:
        user_id = _user_id_for_token(conn, session_token)
        if user_id is None:
            return {}
        rows = conn.execute(
            "SELECT task_id, status, best_time_ms, attempts, solved_at FROM progress WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    return {
        row[0]: ProgressEntry(status=row[1], bestTimeMs=row[2], attempts=row[3], solvedAt=row[4])
        for row in rows
    }


@app.get("/submissions/{task_id}")
def get_submissions(task_id: str, session_token: str | None = Header(default=None, alias="X-Session-Token")) -> list[dict]:
    if not session_token:
        return []
    with _get_db() as conn:
        user_id = _user_id_for_token(conn, session_token)
        if user_id is None:
            return []
        # Newest first; id (not the second-resolution submitted_at) breaks same-second ties
        rows = conn.execute(
            "SELECT id, passed, exec_time_ms, submitted_at, code, mode, note FROM submissions "
            "WHERE user_id = ? AND task_id = ? ORDER BY id DESC LIMIT 50",
            (user_id, task_id)
        ).fetchall()
    return [
        {"id": r[0], "passed": bool(r[1]), "execTimeMs": r[2], "submittedAt": r[3], "code": r[4], "mode": r[5], "note": r[6]}
        for r in rows
    ]


MAX_NOTE_CHARS = 2000


class SubmissionNoteRequest(BaseModel):
    sessionToken: str
    note: str


@app.put("/submissions/{submission_id}/note")
def set_submission_note(submission_id: int, request: SubmissionNoteRequest) -> dict[str, str]:
    note = request.note.strip()
    if len(note) > MAX_NOTE_CHARS:
        raise HTTPException(status_code=400, detail=f"Note is longer than {MAX_NOTE_CHARS} characters")
    with _get_db() as conn:
        # Joining on the session token means a learner can only annotate their own submissions
        cur = conn.execute(
            "UPDATE submissions SET note = ? WHERE id = ? AND user_id = "
            "(SELECT id FROM users WHERE session_token = ?)",
            (note, submission_id, request.sessionToken),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Submission not found")
    return {"note": note}


MAX_DRAFT_CHARS = 200_000


class SaveDraftRequest(BaseModel):
    sessionToken: str
    taskId: str
    mode: str = "core"
    code: str
    updatedAt: int


@app.get("/drafts/{task_id}")
def get_draft(task_id: str, mode: str = "core", session_token: str | None = Header(default=None, alias="X-Session-Token")) -> dict | None:
    if not session_token:
        return None
    with _get_db() as conn:
        user_id = _user_id_for_token(conn, session_token)
        if user_id is None:
            return None
        row = conn.execute(
            "SELECT code, updated_at FROM drafts WHERE user_id = ? AND task_id = ? AND mode = ?",
            (user_id, task_id, mode),
        ).fetchone()
    return {"code": row[0], "updatedAt": row[1]} if row else None


@app.put("/drafts")
def save_draft(request: SaveDraftRequest) -> dict[str, str]:
    if len(request.code) > MAX_DRAFT_CHARS:
        raise HTTPException(status_code=400, detail=f"Draft is longer than {MAX_DRAFT_CHARS} characters")
    with _get_db() as conn:
        # A draft is sometimes a learner's first write, so it may create the user
        user_id = _get_or_create_user(conn, request.sessionToken)
        # A late-arriving older edit (e.g. from another device) never overwrites a newer one
        conn.execute(
            "INSERT INTO drafts (user_id, task_id, mode, code, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (user_id, task_id, mode) DO UPDATE SET code = excluded.code, updated_at = excluded.updated_at "
            "WHERE excluded.updated_at > drafts.updated_at",
            (user_id, request.taskId, request.mode, request.code, request.updatedAt),
        )
    return {"ok": "true"}


MAX_AI_CHAT_CHARS = 1_000_000


class AiChatMessage(BaseModel):
    role: str
    content: str
    label: str | None = None


class SaveAiChatRequest(BaseModel):
    sessionToken: str
    taskId: str
    messages: list[AiChatMessage]


@app.get("/ai-chats/{task_id}")
def get_ai_chat(task_id: str, session_token: str | None = Header(default=None, alias="X-Session-Token")) -> list[dict]:
    if not session_token:
        return []
    with _get_db() as conn:
        user_id = _user_id_for_token(conn, session_token)
        if user_id is None:
            return []
        row = conn.execute(
            "SELECT messages FROM ai_chats WHERE user_id = ? AND task_id = ?", (user_id, task_id)
        ).fetchone()
    return json.loads(row[0]) if row else []


@app.put("/ai-chats")
def save_ai_chat(request: SaveAiChatRequest) -> dict[str, str]:
    messages = json.dumps([m.model_dump(exclude_none=True) for m in request.messages], ensure_ascii=False)
    if len(messages) > MAX_AI_CHAT_CHARS:
        raise HTTPException(status_code=400, detail=f"Chat is longer than {MAX_AI_CHAT_CHARS} characters")
    with _get_db() as conn:
        # Chatting is sometimes a learner's first write, so it may create the user
        user_id = _get_or_create_user(conn, request.sessionToken)
        # Clearing the chat deletes it rather than storing an empty one
        if not request.messages:
            conn.execute("DELETE FROM ai_chats WHERE user_id = ? AND task_id = ?", (user_id, request.taskId))
        else:
            conn.execute(
                "INSERT INTO ai_chats (user_id, task_id, messages) VALUES (?, ?, ?) "
                "ON CONFLICT (user_id, task_id) DO UPDATE SET messages = excluded.messages, updated_at = datetime('now')",
                (user_id, request.taskId, messages),
            )
    return {"ok": "true"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

