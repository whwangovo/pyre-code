#!/usr/bin/env python3
"""Generate Monaco completion data for the Python editor.

Monaco ships a syntax highlighter for Python but no language service, so the
editor can only suggest words that already appear in the buffer -- typing
``np.`` or ``torch.`` offers nothing. This script introspects the exact modules
the grading sandbox injects into every submission's namespace
(see ``grading_service.main._execute_tests``) and writes them to a JSON file the
editor loads lazily.

Run it after upgrading torch/numpy:

    python scripts/gen_completions.py
"""

from __future__ import annotations

import inspect
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "web" / "src" / "lib" / "pythonCompletions.json"

MAX_DOC = 160

# Mirrors the sandbox namespace in grading_service/main.py. Keep in sync.
import numpy  # noqa: E402
import torch  # noqa: E402

NAMESPACES: dict[str, Any] = {
    "torch": torch,
    "nn": torch.nn,
    "F": torch.nn.functional,
    "np": numpy,
    "math": math,
    "Tensor": torch.Tensor,
}

# Offered at the top level, plus after an unrecognised ``name.`` we fall back to
# Tensor members -- in these problems almost every local variable is a tensor.
KEYWORDS = [
    "and", "as", "assert", "break", "class", "continue", "def", "del", "elif",
    "else", "except", "False", "finally", "for", "from", "global", "if",
    "import", "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass",
    "raise", "return", "True", "try", "while", "with", "yield",
]
BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "int", "isinstance", "len", "list", "map", "max", "min",
    "print", "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "type", "zip",
]

_SIG_LINE = re.compile(r"^\s*\w+\((.*?)\)(\s*->\s*[^\n]+)?\s*$")


def _skip_signature_block(lines: list[str], name: str) -> list[str]:
    """Drop the leading ``name(...)`` signature, which torch and numpy docstrings
    both carry and numpy wraps across several lines."""
    out = [ln for ln in lines if ln]
    while out:
        first = out[0]
        if _SIG_LINE.match(first) and first.startswith(f"{name}("):
            out = out[1:]
            continue
        if first.startswith(f"{name}("):
            # a multi-line signature: consume until the parentheses balance
            depth = 0
            for i, ln in enumerate(out):
                depth += ln.count("(") - ln.count(")")
                if depth <= 0:
                    out = out[i + 1:]
                    break
            else:
                return []
            continue
        break
    return out


def _signature(obj: Any, name: str) -> str:
    """Best-effort signature. C extensions defeat inspect, so fall back to the
    first docstring line, which torch and numpy both format as a signature."""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        pass
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines()[:3]:
        line = line.strip()
        if line.startswith(f"{name}(") and _SIG_LINE.match(line):
            return line[len(name):]
    return "(...)"


def _summary(obj: Any, name: str) -> str:
    doc = inspect.getdoc(obj) or ""
    lines = [ln.strip() for ln in doc.splitlines()]
    body = _skip_signature_block(lines, name)
    if not body:
        return ""
    text = body[0]
    if len(text) > MAX_DOC:
        text = text[: MAX_DOC - 1].rstrip() + "…"
    return text


def _value(obj: Any) -> str:
    """Short repr for a constant. Never call getdoc on one: inspect walks up to
    the *type*, so math.pi would document itself as float.__doc__."""
    try:
        text = repr(obj)
    except Exception:
        return ""
    text = " ".join(text.split())
    if len(text) > 60:
        return ""
    return text


def _kind(obj: Any) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    if inspect.isdatadescriptor(obj):
        # Tensor.shape and friends: a descriptor, not a value. Its own __doc__ is
        # meaningful, and repr() would print "<attribute 'shape' of ...>".
        return "property"
    return "constant"


def collect(module: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        kind = _kind(obj)
        entry: dict[str, str] = {"l": name, "k": kind}
        if kind in ("function", "class", "property"):
            if kind != "property":
                sig = _signature(obj, name)
                if sig and sig != "(...)":
                    entry["d"] = sig
            summary = _summary(obj, name)
            if summary:
                entry["s"] = summary
        elif kind == "constant":
            value = _value(obj)
            if value:
                entry["d"] = value
        out.append(entry)
    return out


def main() -> None:
    data = {
        "_comment": "Generated by scripts/gen_completions.py -- do not edit by hand.",
        "versions": {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "keywords": KEYWORDS,
        "builtins": BUILTINS,
        "namespaces": {alias: collect(mod) for alias, mod in NAMESPACES.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n",
                      encoding="utf-8")
    total = sum(len(v) for v in data["namespaces"].values())
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {total} completions across {len(NAMESPACES)} namespaces "
          f"to {OUTPUT.relative_to(ROOT)} ({size_kb:.0f} KB)")
    for alias, entries in data["namespaces"].items():
        print(f"  {alias:8s} {len(entries):5d}")


if __name__ == "__main__":
    main()
