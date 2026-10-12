"""Deterministic assertions, cheapest first. No model calls, no network.

`run_checks(case, result)` takes one golden case and one model result and
returns a list of CheckResult. Which checks run depends on the case's
`expect` block:

  {"json":  {"category": ..., "priority": ..., ...}}   JSON routes
  {"tools": [{"name": ..., "args": {...}}, ...]}       tool-calling routes
                                                       ([] = answered without tools)

Two kinds of check are mixed in here on purpose. Baseline checks (`fields`,
`tool_set`, `tool_args`) compare the candidate against what production did.
Contract checks (`json_schema`, `arg_format`, `summary_length`) encode what the
prompt promises and hold regardless of what production did.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


# --- contract tables -------------------------------------------------------

# Required keys of the triage JSON and their allowed values (None = free text).
JSON_SCHEMA: dict[str, set[str] | None] = {
    "category": {"billing", "bug", "feature_request", "account", "other"},
    "priority": {"low", "medium", "high", "urgent"},
    "sentiment": {"negative", "neutral", "positive"},
    "summary": None,
    "suggested_reply": None,
}
SUMMARY_MAX_WORDS = 20

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_CURRENCY = {"USD", "EUR", "GBP", "JPY", "CHF"}


def _is_iso_date(value: object) -> bool:
    return isinstance(value, str) and bool(ISO_DATE.match(value))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# What the prompts promise the tools will receive. The model is supposed to
# normalise before calling; these rules say whether it did.
ARG_RULES = {
    "get_schedule": {"date": _is_iso_date},
    "days_between": {"start_date": _is_iso_date, "end_date": _is_iso_date},
    "convert_currency": {
        "from_currency": lambda v: v in ISO_CURRENCY,
        "to_currency": lambda v: v in ISO_CURRENCY,
    },
    "convert_units": {"value": _is_number},
}

# Arguments the model phrases freely. A different wording is drift worth
# printing, not a failure.
FREE_TEXT_ARGS = {("search_notes", "query")}


# --- entry point -----------------------------------------------------------


def run_checks(case: dict, result: dict) -> list[CheckResult]:
    expect = case["expect"]
    out: list[CheckResult] = []
    if "json" in expect:
        out += json_checks(result["output"], expect["json"])
    if "tools" in expect:
        out += tool_checks(result, expect["tools"])
    return out


# --- JSON routes -----------------------------------------------------------


def strip_fences(text: str) -> str:
    """Lenient: drop a ```json fence if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def json_checks(output: str, expected_fields: dict) -> list[CheckResult]:
    results: list[CheckResult] = []

    try:
        json.loads(output)
        results.append(CheckResult("json_strict", True))
    except json.JSONDecodeError as exc:
        why = "wrapped in markdown fences" if output.strip().startswith("```") else str(exc)[:60]
        results.append(CheckResult("json_strict", False, why))

    try:
        data = json.loads(strip_fences(output))
    except json.JSONDecodeError:
        for name in ("json_schema", "fields", "summary_length"):
            results.append(CheckResult(name, False, "no JSON object found"))
        return results
    if not isinstance(data, dict):
        for name in ("json_schema", "fields", "summary_length"):
            results.append(CheckResult(name, False, "JSON is not an object"))
        return results

    problems = []
    for key, allowed in JSON_SCHEMA.items():
        if key not in data:
            problems.append(f"missing {key}")
        elif allowed is not None and data[key] not in allowed:
            problems.append(f"{key}={data[key]!r} not in {sorted(allowed)}")
    results.append(CheckResult("json_schema", not problems, "; ".join(problems)))

    diffs = [
        f"{key}: {expected!r} -> {data.get(key)!r}"
        for key, expected in expected_fields.items()
        if data.get(key) != expected
    ]
    results.append(CheckResult("fields", not diffs, "; ".join(diffs)))

    words = len(str(data.get("summary", "")).split())
    results.append(CheckResult("summary_length", words <= SUMMARY_MAX_WORDS, f"{words} words"))
    return results


# --- tool-calling routes ---------------------------------------------------


def tool_checks(result: dict, expected_calls: list[dict]) -> list[CheckResult]:
    calls = result["tool_calls"]
    results: list[CheckResult] = []

    got = {c["name"] for c in calls}
    want = {c["name"] for c in expected_calls}
    extra, missing = sorted(got - want), sorted(want - got)
    parts = []
    if extra:
        parts.append(f"extra {extra}")
    if missing:
        parts.append(f"missing {missing}")
    results.append(CheckResult("tool_set", not parts, "; ".join(parts)))

    if not expected_calls:
        # Production answered in text. The candidate must too.
        answered = bool(result["output"].strip())
        results.append(CheckResult("answered", answered, "" if answered else "no text reply"))
        return results

    failures, drift = [], []
    for exp in expected_calls:
        same_name = [c for c in calls if c["name"] == exp["name"]]
        if not same_name:
            failures.append(f"{exp['name']} not called")
            continue
        best = min(same_name, key=lambda c: _mismatches(exp["args"], c["args"]))
        for field, value in exp["args"].items():
            actual = best["args"].get(field)
            if actual == value:
                continue
            message = f"{exp['name']}.{field}: {value!r} -> {actual!r}"
            if (exp["name"], field) in FREE_TEXT_ARGS:
                drift.append(f"(drift) {message}")
            else:
                failures.append(message)
    results.append(CheckResult("tool_args", not failures, "; ".join(failures + drift)))

    bad = []
    for call in calls:
        for field, ok in ARG_RULES.get(call["name"], {}).items():
            if field in call["args"] and not ok(call["args"][field]):
                bad.append(f"{call['name']}.{field}={call['args'][field]!r}")
    results.append(CheckResult("arg_format", not bad, "; ".join(bad)))
    return results


def _mismatches(expected: dict, actual: dict) -> int:
    return sum(1 for key, value in expected.items() if actual.get(key) != value)
