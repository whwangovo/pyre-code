"""Algorithm-problem harness: value types, ACM text I/O, answer checkers, core-mode runner.

Stdlib only and self-contained, because it serves two roles:
- imported by the grading service (ACM serialization, checking, display);
- executed as ``python -I runner.py`` for core mode, reading a JSON job on stdin and
  printing a JSON result as the last stdout line. ``-I`` keeps the learner's code from
  picking up anything on sys.path, so this file cannot import project modules.

Type names used in task specs:
    int, float, bool, str, int[], int[][], str[], str[][], char[][], ListNode, ListNode[],
    TreeNode, Node (list node with a random pointer), void
A trailing ``?`` (``int?``) marks a scalar answer that may be null.
"""

from __future__ import annotations

import io
import json
import sys
import time
import traceback
from collections import deque
from typing import Any, List, Optional

MAX_DISPLAY_CHARS = 2000

# Limits for the sandbox subprocess (applied in ``_apply_limits``). Hot 100 answers never
# need more than a fraction of this; the caps turn a runaway program into a per-case error.
MEMORY_LIMIT_BYTES = 1 << 30  # 1 GiB
CPU_LIMIT_S = 4  # CPU seconds; the 3 s wall-clock timeout in judge.py normally fires first
CPU_LIMIT_HARD_S = 5


def _apply_limits() -> None:
    """Cap this subprocess's memory and CPU from inside it.

    Set here rather than via ``preexec_fn``, because forking a threaded FastAPI process
    with callbacks is unsafe. Only this runner applies RLIMIT_AS: it is stdlib-only,
    unlike the torch-based grading-service runner whose CUDA reservation would trip it.
    """
    try:
        import resource
    except ImportError:  # Windows has no rlimits; there the container limit is the guard
        return
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_S, CPU_LIMIT_HARD_S))


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    """Linked-list node with an extra pointer to any node of the list, or None."""

    def __init__(self, x: int, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random


# ---------------------------------------------------------------------------
# Plain values <-> runtime objects (LeetCode-style array encodings)
# ---------------------------------------------------------------------------

def build_list(values: list) -> Optional[ListNode]:
    dummy = tail = ListNode()
    for v in values:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def list_to_array(head: Optional[ListNode], limit: int = 10**6) -> list:
    out = []
    while head is not None:
        out.append(head.val)
        if len(out) > limit:
            raise ValueError("linked list too long (cycle?)")
        head = head.next
    return out


def build_random_list(pairs: list) -> Optional[Node]:
    """``[[val, random_index or None], ...]`` -> head."""
    nodes = [Node(val) for val, _ in pairs]
    for i, (_, rand) in enumerate(pairs):
        if i + 1 < len(nodes):
            nodes[i].next = nodes[i + 1]
        nodes[i].random = None if rand is None else nodes[rand]
    return nodes[0] if nodes else None


def random_list_to_array(head: Optional[Node], limit: int = 10**6) -> list:
    nodes = []
    while head is not None:
        nodes.append(head)
        if len(nodes) > limit:
            raise ValueError("linked list too long (cycle?)")
        head = head.next
    index = {id(n): i for i, n in enumerate(nodes)}
    out = []
    for n in nodes:
        if n.random is not None and id(n.random) not in index:
            raise ValueError("a random pointer leaves the list")
        out.append([n.val, None if n.random is None else index[id(n.random)]])
    return out


def build_tree(values: list) -> Optional[TreeNode]:
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def tree_to_array(root: Optional[TreeNode]) -> list:
    out: list = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node is None:
            out.append(None)
            continue
        out.append(node.val)
        queue.append(node.left)
        queue.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out


def to_runtime(value: Any, type_name: str) -> Any:
    if type_name == "ListNode":
        return build_list(value)
    if type_name == "ListNode[]":
        return [build_list(v) for v in value]
    if type_name == "TreeNode":
        return build_tree(value)
    if type_name == "Node":
        return build_random_list(value)
    return json.loads(json.dumps(value))  # deep copy; the solution may mutate its inputs


def from_runtime(value: Any, type_name: str) -> Any:
    if type_name == "ListNode":
        return list_to_array(value)
    if type_name == "TreeNode":
        return tree_to_array(value)
    if type_name == "Node":
        return random_list_to_array(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [list(v) if isinstance(v, tuple) else v for v in value]
    return value


# ---------------------------------------------------------------------------
# Display (LeetCode style: JSON with null)
# ---------------------------------------------------------------------------

def _clip(text: str) -> str:
    return text if len(text) <= MAX_DISPLAY_CHARS else text[:MAX_DISPLAY_CHARS] + " …"


def show(value: Any) -> str:
    return _clip(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def show_args(params: list, args: list) -> str:
    return _clip(", ".join(f"{name} = {show(arg)}" for (name, _), arg in zip(params, args)))


# ---------------------------------------------------------------------------
# ACM text format
# ---------------------------------------------------------------------------

def _scalar_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _row_text(values: list) -> str:
    return " ".join("null" if v is None else _scalar_text(v) for v in values)


def acm_input_lines(type_name: str, value: Any) -> list[str]:
    """Lines one argument occupies in ACM stdin."""
    if type_name in ("int", "float", "bool", "str"):
        return [_scalar_text(value)]
    if type_name in ("int[]", "ListNode"):
        return [str(len(value)), _row_text(value)]
    if type_name == "int[][]":
        return [str(len(value))] + [_row_text(row) for row in value]
    if type_name == "char[][]":
        return [str(len(value))] + ["".join(row) for row in value]
    if type_name == "str[]":
        return [str(len(value))] + list(value)
    if type_name == "ListNode[]":
        return [str(len(value))] + [line for v in value for line in acm_input_lines("ListNode", v)]
    if type_name == "Node":
        return [str(len(value))] + [_row_text(pair) for pair in value]
    if type_name == "TreeNode":
        return [_row_text(value)]
    raise NotImplementedError(f"ACM input for type {type_name}")


def acm_output_text(type_name: str, value: Any) -> str:
    """Canonical ACM stdout for a return value (without trailing newline)."""
    type_name = type_name.rstrip("?")
    if type_name in ("int", "float", "bool", "str"):
        return _scalar_text(value)
    if type_name in ("int[]", "ListNode", "TreeNode"):
        return _row_text(value)
    if type_name in ("int[][]", "str[][]", "Node"):
        return "\n".join(_row_text(row) for row in value)
    if type_name == "char[][]":
        return "\n".join("".join(row) for row in value)
    if type_name == "str[]":
        return "\n".join(value)
    raise NotImplementedError(f"ACM output for type {type_name}")


def _parse_token(token: str, cast) -> Any:
    return None if token == "null" else cast(token)


def parse_acm_output(type_name: str, text: str, keep_empty_rows: bool = False) -> Any:
    """Inverse of acm_output_text, tolerant of trailing whitespace. Raises ValueError.

    Trailing blank lines are dropped, unless ``keep_empty_rows``: then only the final newline
    is, so a nested answer can end with an empty row (``[[1], []]`` prints ``1``, then ``""``)."""
    text = text.replace("\r\n", "\n")
    if keep_empty_rows:
        text = text[:-1] if text.endswith("\n") else text
        lines = [line.rstrip() for line in text.split("\n")] if text else []
    else:
        lines = [line.rstrip() for line in text.split("\n")]
        while lines and lines[-1] == "":
            lines.pop()
    nullable = type_name.endswith("?")
    type_name = type_name.rstrip("?")
    if nullable and type_name != "str" and [line.strip() for line in lines] == ["null"]:
        return None
    if type_name in ("int", "float", "bool", "str"):
        if type_name == "str":
            if len(lines) > 1:
                raise ValueError("expected a single line")
            return lines[0] if lines else ""
        if len(lines) != 1:
            raise ValueError("expected exactly one line")
        token = lines[0].strip()
        if type_name == "int":
            return int(token)
        if type_name == "float":
            return float(token)
        if token not in ("true", "false"):
            raise ValueError("expected true or false")
        return token == "true"
    if type_name in ("int[]", "ListNode"):
        if len(lines) > 1:
            raise ValueError("expected a single line")
        return [int(t) for t in lines[0].split()] if lines else []
    if type_name == "TreeNode":
        if len(lines) > 1:
            raise ValueError("expected a single line")
        return [_parse_token(t, int) for t in lines[0].split()] if lines else []
    if type_name == "int[][]":
        return [[int(t) for t in line.split()] for line in lines]
    if type_name == "Node":
        return [[_parse_token(t, int) for t in line.split()] for line in lines]
    if type_name == "str[][]":
        return [line.split() for line in lines]
    if type_name == "char[][]":
        return [list(line) for line in lines]
    if type_name == "str[]":
        return lines
    raise NotImplementedError(f"ACM output for type {type_name}")


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------

def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _approx_equal(actual: Any, expected: Any) -> bool:
    """Equality where float leaves compare with a tolerance (design problems that return floats)."""
    if isinstance(expected, float) and not isinstance(actual, bool):
        return check("float", actual, expected)
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_approx_equal(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def check(kind: str, actual: Any, expected: Any) -> bool:
    if kind == "exact":
        return actual == expected
    if kind == "approx":
        return _approx_equal(actual, expected)
    if kind == "unordered":
        return isinstance(actual, list) and sorted(actual, key=_sort_key) == sorted(expected, key=_sort_key)
    if kind == "unordered_nested":
        if not isinstance(actual, list) or not all(isinstance(g, list) for g in actual):
            return False
        norm = lambda groups: sorted((sorted(g, key=_sort_key) for g in groups), key=_sort_key)
        return norm(actual) == norm(expected)
    if kind == "float":
        return isinstance(actual, (int, float)) and abs(actual - expected) <= 1e-5 * max(1.0, abs(expected))
    raise ValueError(f"unknown checker {kind}")


# Task-supplied hooks (prepare / finish / checker) arrive as source strings, since this file
# can't import task modules. They see the helpers below.
HOOK_GLOBALS = {
    "ListNode": ListNode, "TreeNode": TreeNode,
    "build_list": build_list, "list_to_array": list_to_array,
    "build_tree": build_tree, "tree_to_array": tree_to_array,
    "Node": Node, "build_random_list": build_random_list, "random_list_to_array": random_list_to_array,
}
_hook_cache: dict[str, dict] = {}


def hook(source: str, name: str):
    ns = _hook_cache.get(source)
    if ns is None:
        ns = _hook_cache[source] = dict(HOOK_GLOBALS)
        exec(source, ns)
    return ns[name]


def check_answer(spec: dict, args: Any, actual: Any, expected: Any) -> bool:
    """Compare a plain answer with the expected one. ``custom`` tasks validate it with their
    own ``check(args, actual, expected)``, for problems with several valid answers."""
    kind = spec.get("check", "exact")
    if kind == "custom":
        try:
            return bool(hook(spec["checker"], "check")(args, actual, expected))
        except Exception:  # noqa: BLE001 - a malformed answer is just wrong
            return False
    return check(kind, actual, expected)


# ---------------------------------------------------------------------------
# Core-mode execution (runs inside the sandbox subprocess)
# ---------------------------------------------------------------------------

def _user_error(exc: BaseException) -> str:
    """Exception message with the innermost line number of the learner's code."""
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<solution>":
            line = frame.lineno
    where = f" (line {line})" if line else ""
    return f"{type(exc).__name__}: {exc}{where}"


def solution_namespace(name: str = "__solution__") -> dict[str, Any]:
    """Globals for core-mode code: what LeetCode predefines."""
    return {"__name__": name, "List": List, "Optional": Optional, "ListNode": ListNode, "TreeNode": TreeNode, "Node": Node}


def compute(ns: dict, spec: dict, case: dict) -> Any:
    """Run one case against loaded code and return the plain (JSON-able) answer."""
    if spec.get("design"):
        return _run_design(ns, spec["design"], case)
    return _run_method(ns, spec, case)


def run_core(job: dict) -> list[dict]:
    """Load the learner's code once, then run each of ``job["cases"]`` in order."""
    spec = job["spec"]
    # Like LeetCode: recursive solutions over 10^3–10^4 deep inputs shouldn't hit Python's default 1000
    sys.setrecursionlimit(20000)
    ns = solution_namespace()
    real_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        exec(compile(job["code"], "<solution>", "exec"), ns)
    except BaseException as exc:  # noqa: BLE001 - report anything the learner's code raises
        error = {"status": "error", "error": _user_error(exc), "stdout": captured.getvalue()}
        return [error for _ in job["cases"]]
    finally:
        sys.stdout = real_stdout

    results = []
    for case in job["cases"]:
        sys.stdout = captured = io.StringIO()
        start = time.perf_counter()
        try:
            actual = compute(ns, spec, case)
        except BaseException as exc:  # noqa: BLE001
            results.append({"status": "error", "error": _user_error(exc), "stdout": captured.getvalue()})
            continue
        finally:
            sys.stdout = real_stdout
        elapsed_ms = (time.perf_counter() - start) * 1000
        passed = check_answer(spec, case.get("args"), actual, case["expected"])
        results.append({
            "status": "ok" if passed else "wrong",
            "actual": show(actual),
            "timeMs": elapsed_ms,
            "stdout": captured.getvalue(),
        })
    return results


def _run_method(ns: dict, spec: dict, case: dict) -> Any:
    solution_cls = ns.get("Solution")
    if solution_cls is None:
        raise NameError("class Solution is not defined")
    method = getattr(solution_cls(), spec["function_name"], None)
    if method is None:
        raise AttributeError(f"Solution has no method '{spec['function_name']}'")
    params = spec["params"]
    # Case args follow ``inputs`` (the plain data); ``prepare`` turns them into the method's
    # arguments when the two differ (shared list tails, cycles, nodes picked by value)
    args = [to_runtime(arg, type_name) for (_, type_name), arg in zip(spec["inputs"], case["args"])]
    if spec.get("prepare"):
        args = list(hook(spec["prepare"], "prepare")(args))
    result = method(*args)
    if spec.get("finish"):
        return hook(spec["finish"], "finish")(result, args)
    inplace = spec.get("inplace")
    if inplace is not None:
        index = [name for name, _ in params].index(inplace)
        return from_runtime(args[index], params[index][1])
    return from_runtime(result, spec["returns"])


def _run_design(ns: dict, design: dict, case: dict) -> list:
    cls = ns.get(design["class"])
    if cls is None:
        raise NameError(f"class {design['class']} is not defined")
    ops, op_args = case["ops"], case["args"]
    obj = cls(*op_args[0])
    out: list = [None]
    for op, args in zip(ops[1:], op_args[1:]):
        result = getattr(obj, op)(*args)
        out.append(None if design["methods"][op][1] == "void" else result)
    return out


if __name__ == "__main__":
    _apply_limits()
    results = run_core(json.loads(sys.stdin.read()))
    sys.stdout.write("\n" + json.dumps(results) + "\n")
