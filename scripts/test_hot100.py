"""Tests for the Hot 100 judge: every reference solution passes in both modes, and the
grader rejects wrong, slow, oversized and malformed answers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge.algo import judge, runner, spec
from torch_judge.tasks import HOT100_TASKS


def _grade(task_id, code, mode="core", indices=None):
    return judge.grade(task_id, HOT100_TASKS[task_id], code, mode, indices)


@pytest.mark.parametrize("mode", spec.MODES)
@pytest.mark.parametrize("task_id", sorted(HOT100_TASKS))
def test_reference_solution_passes(task_id, mode):
    task = HOT100_TASKS[task_id]
    code = task["solution"] if mode == "core" else spec.acm_solution(task)
    result = _grade(task_id, code, mode)
    failures = [(r["name"], r["error"]) for r in result["results"] if not r["passed"]]
    assert result["allPassed"], failures


@pytest.mark.parametrize("task_id", sorted(HOT100_TASKS))
def test_task_has_two_visible_samples_and_a_large_case(task_id):
    for mode in spec.MODES:
        cases = spec.cases_for_mode(task_id, HOT100_TASKS[task_id], mode)
        assert len(cases) >= 3
        assert not any(c.get("hidden") for c in cases[:2])


def test_brute_force_times_out_on_large_input():
    brute = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        for i in range(len(nums)):\n"
        "            for j in range(i + 1, len(nums)):\n"
        "                if nums[i] + nums[j] == target:\n"
        "                    return [i, j]\n"
    )
    result = _grade("two_sum", brute)
    verdicts = {r["name"]: r for r in result["results"]}
    assert all(r["passed"] for name, r in verdicts.items() if name != "Large input")
    assert "Time Limit Exceeded" in verdicts["Large input"]["error"]


def test_core_memory_bomb_becomes_a_memory_error():
    # ~2 GiB of pointers: the runner's 1 GiB RLIMIT_AS makes the allocation fail at once.
    # Deliberately not larger, so a regression fails this test before it can hurt the machine.
    code = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        x = [0] * (2**28)\n"
        "        return [0, 1]\n"
    )
    result = _grade("two_sum", code)
    assert result["total"] > 0 and not result["allPassed"]
    assert all("MemoryError" in (r["error"] or "") for r in result["results"])


def test_acm_memory_bomb_is_contained():
    code = "data = bytearray(2**31)\nprint('0')\n"
    result = _grade("two_sum", code, mode="acm")
    assert result["total"] > 0 and not result["allPassed"]
    assert all("Runtime Error" in (r["error"] or "") for r in result["results"])


# Classic wrong solutions, one file per task (named like the task); each must fail somewhere
# in the hand-written or random cases
MUTANTS_DIR = Path(__file__).parent / "hot100_mutants"
MIN_RANDOM_CATCHES = 4
MUTANTS = {p.stem: p.read_text(encoding="utf-8") for p in MUTANTS_DIR.glob("*.py")}


def test_every_task_has_a_mutant():
    assert sorted(HOT100_TASKS) == sorted(MUTANTS)


@pytest.mark.parametrize("task_id", sorted(MUTANTS))
def test_classic_mistake_is_caught_by_random_cases_too(task_id):
    # A margin, not just one lucky case: random cases should expose the mistake often enough
    # that similar bugs are caught too (README: retune random_case if this fails)
    failed = [r["name"] for r in _grade(task_id, MUTANTS[task_id])["results"] if not r["passed"]]
    caught = sum(name.startswith("Random") for name in failed)
    assert caught >= MIN_RANDOM_CATCHES, f"{caught} of {spec.RANDOM_CASES} random cases catch it: {failed}"


def test_random_cases_are_seeded_and_visible():
    task = HOT100_TASKS["two_sum"]
    first = spec.cases_for_mode("two_sum", task, "core")
    again = spec.cases_for_mode("two_sum", task, "core")
    randoms = [c for c in first if c["name"].startswith("Random")]
    assert len(randoms) == spec.RANDOM_CASES and not any(c.get("hidden") for c in randoms)
    assert first == again


def test_hung_case_in_batch_is_pinned_to_that_case():
    # Hangs only on two-element inputs; the batch times out, then per-case reruns find them
    code = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        while len(nums) == 2:\n"
        "            pass\n"
        "        s = {}\n"
        "        for i, x in enumerate(nums):\n"
        "            if target - x in s:\n"
        "                return [s[target - x], i]\n"
        "            s[x] = i\n"
    )
    cases = spec.cases_for_mode("two_sum", HOT100_TASKS["two_sum"], "core")
    results = _grade("two_sum", code)["results"]
    for case, result in zip(cases, results):
        two_elements = not case.get("hidden") and len(case["args"][0]) == 2
        assert result["passed"] != two_elements, case["name"]


def test_wrong_answer_reports_actual_value():
    result = _grade("coin_change", "class Solution:\n    def coinChange(self, coins, amount):\n        return 7\n", indices=[0])
    case = result["results"][0]
    assert not case["passed"] and case["error"] == "Wrong Answer"
    assert case["actual"] == "7" and case["expected"] == "2"


def test_inplace_problem_requires_mutation():
    returns_copy = "class Solution:\n    def moveZeroes(self, nums):\n        return sorted(nums, key=lambda x: x == 0)\n"
    assert not _grade("move_zeroes", returns_copy, indices=[0])["allPassed"]


def test_runtime_error_points_at_learner_line():
    result = _grade("coin_change", "class Solution:\n    def coinChange(self, coins, amount):\n        return coins[99]\n", indices=[0])
    assert result["results"][0]["error"] == "IndexError: list index out of range (line 3)"


def test_acm_accepts_any_order_for_unordered_answers():
    assert _grade("two_sum", "print(3, 2)\n", "acm", [0])["allPassed"]


def test_acm_rejects_malformed_output():
    case = _grade("two_sum", "print('[2, 3]')\n", "acm", [0])["results"][0]
    assert not case["passed"] and case["error"].startswith("Wrong Answer (output format")


def test_acm_runtime_error_hides_temp_path():
    case = _grade("two_sum", "print(1 / 0)\n", "acm", [0])["results"][0]
    assert "ZeroDivisionError" in case["error"] and 'File "solution.py"' in case["error"]


@pytest.mark.parametrize("type_name, value", [
    ("int[]", [3, -1, 0]),
    ("int[]", []),
    ("int[][]", [[5], [3, 8], [4, 7, 9]]),
    ("TreeNode", [5, 3, 8, None, 4]),
    ("bool", True),
    ("str[]", ["ab", "c"]),
])
def test_acm_output_round_trips(type_name, value):
    assert runner.parse_acm_output(type_name, runner.acm_output_text(type_name, value) + "\n") == value


def test_tree_codec_round_trips():
    values = [1, 2, 3, None, 4, None, 5, 6]
    assert runner.tree_to_array(runner.build_tree(values)) == values


# ---------------------------------------------------------------------------
# Framework types without a shipped problem of their own yet: small synthetic tasks, graded
# with their reference in both modes
# ---------------------------------------------------------------------------

def _synthetic(params, returns, solution, cases, **extra):
    return {"function_name": "f", "params": params, "returns": returns, "solution": solution,
            "cases": cases, **extra}


SYNTHETIC = {
    "char_grid": _synthetic(
        [("grid", "char[][]")], "char[][]",
        "class Solution:\n    def f(self, grid):\n        return [list(r) for r in zip(*grid)]\n",
        [{"args": [[["1", "0"], ["#", "."]]], "expected": [["1", "#"], ["0", "."]]},
         {"args": [[["a", "b", "c"]]], "expected": [["a"], ["b"], ["c"]]}],
    ),
    "list_of_lists": _synthetic(
        [("lists", "ListNode[]")], "ListNode",
        "class Solution:\n    def f(self, lists):\n        vals = []\n        for h in lists:\n"
        "            while h:\n                vals.append(h.val)\n                h = h.next\n"
        "        dummy = t = ListNode()\n        for v in sorted(vals):\n            t.next = ListNode(v)\n            t = t.next\n"
        "        return dummy.next\n",
        [{"args": [[[1, 4], [], [2, 3]]], "expected": [1, 2, 3, 4]},
         {"args": [[]], "expected": []}],
    ),
    "random_pointers": _synthetic(
        [("head", "Node")], "Node",
        "class Solution:\n    def f(self, head):\n        copies = {None: None}\n        n = head\n"
        "        while n:\n            copies[n] = Node(n.val)\n            n = n.next\n        n = head\n"
        "        while n:\n            copies[n].next, copies[n].random = copies[n.next], copies[n.random]\n            n = n.next\n"
        "        return copies[head]\n",
        [{"args": [[[7, None], [3, 0], [5, 1]]], "expected": [[7, None], [3, 0], [5, 1]]},
         {"args": [[]], "expected": []}],
    ),
    "nullable_answer": _synthetic(
        [("nums", "int[]")], "int",
        "class Solution:\n    def f(self, nums):\n        return max(nums) if nums else None\n",
        [{"args": [[3, 9, 2]], "expected": 9}, {"args": [[]], "expected": None}],
        output="int?",
    ),
    "float_design": {
        "function_name": "Avg",
        "design": {"class": "Avg", "init": {}, "methods": {"add": ({"x": "int"}, "void"), "mean": ({}, "float")}},
        "solution": "class Avg:\n    def __init__(self):\n        self.xs = []\n"
                    "    def add(self, x):\n        self.xs.append(x)\n    def mean(self):\n        return sum(self.xs) / len(self.xs)\n",
        "cases": [{"ops": ["Avg", "add", "add", "mean", "add", "mean"], "args": [[], [1], [2], [], [4], []],
                   "expected": [None, None, None, 1.5, None, 7 / 3]}],
    },
}


@pytest.mark.parametrize("mode", spec.MODES)
@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_framework_type_round_trips(name, mode):
    task = SYNTHETIC[name]
    code = task["solution"] if mode == "core" else spec.acm_solution(task)
    result = judge.grade(name, task, code, mode)
    assert result["allPassed"], [(r["name"], r["error"], r.get("actual")) for r in result["results"]]


def test_design_float_output_compares_numerically():
    task = SYNTHETIC["float_design"]
    assert judge.grade("float_design", task, "print('1.500000')\nprint(2.3333333)\n", "acm")["allPassed"]
    assert not judge.grade("float_design", task, "print(1.5)\nprint(2.4)\n", "acm")["allPassed"]


def test_custom_checker_accepts_any_valid_answer():
    # "abcd": every single character is a longest palindrome
    case = spec.cases_for_mode("longest_palindromic_substring", HOT100_TASKS["longest_palindromic_substring"], "core")
    index = next(i for i, c in enumerate(case) if c["args"] == ["abcd"])
    returns_d = "class Solution:\n    def longestPalindrome(self, s):\n        return s[-1]\n"
    returns_two = "class Solution:\n    def longestPalindrome(self, s):\n        return s[:2]\n"
    assert _grade("longest_palindromic_substring", returns_d, indices=[index])["allPassed"]
    assert not _grade("longest_palindromic_substring", returns_two, indices=[index])["allPassed"]
    assert _grade("longest_palindromic_substring", "print(input()[1])\n", "acm", [index])["allPassed"]


def test_finish_hook_rejects_a_copied_node():
    copy = "class Solution:\n    def getIntersectionNode(self, headA, headB):\n        return ListNode(6)\n"
    case = _grade("intersection_of_two_linked_lists", copy, indices=[0])["results"][0]
    assert not case["passed"] and "not in both lists" in case["actual"]
