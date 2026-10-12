"""The runner, on the committed Sonnet 5 -> Sonnet 5.5 results and on synthetic samples."""

import subprocess
import sys

from run_eval import pass_rates, score

CASES = [
    {"id": "a", "route": "planner", "expect": {"tools": [{"name": "get_tasks", "args": {}}]}},
    {"id": "b", "route": "planner", "expect": {"tools": [{"name": "get_tasks", "args": {}}]}},
]
GOOD = {"output": "", "tool_calls": [{"name": "get_tasks", "args": {}}]}
BAD = {"output": "", "tool_calls": [{"name": "get_schedule", "args": {"date": "2026-09-08"}}]}


def test_repeats_turn_a_flaky_case_into_a_fraction():
    samples = {"a": [GOOD, BAD], "b": [GOOD, GOOD]}
    scores = score(CASES, samples)
    assert scores["a"]["tool_set"] == 0.5
    assert scores["b"]["tool_set"] == 1.0
    rates = pass_rates(CASES, scores)
    assert rates[("planner", "tool_set")] == (0.75, 2)


def test_committed_results_reproduce_the_post():
    proc = subprocess.run(
        [sys.executable, "run_eval.py"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = proc.stdout
    assert "planner-000" in out and "get_schedule.date: '2026-09-08' -> 'today'" in out
    assert "triage-003" in out and "priority: 'urgent' -> 'high'" in out
    assert "triage-006" in out and "category: 'billing' -> 'bug'" in out
    assert "converter-008" in out and "extra ['convert_units']" in out
    assert "json_strict" in out and "improved" in out
    assert out.rstrip().endswith("points)")
    assert "FAIL: 6 check(s) dropped more than 2%" in out


def test_a_generous_threshold_passes_the_same_data():
    proc = subprocess.run(
        [sys.executable, "run_eval.py", "--max-drop", "1.0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout
