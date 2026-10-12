"""Turn production logs into a golden set.

    python build_golden.py logs/production.jsonl > golden.jsonl

One log line per model turn, as your app would write it:

    {"request_id": "...", "ts": "...", "route": "triage", "history": [...],
     "input": "...", "output": "...", "tool_calls": [{"name": ..., "args": {...}}],
     "tags": ["edge"]}           # tags are optional

What this does, in order:
  1. strips PII from inputs and history
  2. drops repeated inputs (they would make the suite look more certain than it is)
  3. samples up to --per-route lines per route, oldest first, always keeping
     lines tagged "edge"
  4. writes an `expect` block derived from what production did: the tools it
     called with their arguments, or the enum fields of its JSON output

Review the result before committing it. The expectations say "do what
production did", and production is a baseline, not an oracle.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

from checks import JSON_SCHEMA, strip_fences

JSON_ROUTES = {"triage"}

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "[phone]"),
]


def strip_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def expectation(line: dict) -> dict:
    if line["route"] in JSON_ROUTES:
        data = json.loads(strip_fences(line["output"]))
        enum_fields = [key for key, allowed in JSON_SCHEMA.items() if allowed is not None]
        return {"json": {key: data[key] for key in enum_fields}}
    return {"tools": [{"name": c["name"], "args": c["args"]} for c in line["tool_calls"]]}


def build(lines: list[dict], per_route: int) -> list[dict]:
    by_route: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for line in sorted(lines, key=lambda x: x["ts"]):
        key = (line["route"], line["input"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        by_route[line["route"]].append(line)

    cases = []
    for route, route_lines in sorted(by_route.items()):
        edge = [x for x in route_lines if "edge" in x.get("tags", [])]
        rest = [x for x in route_lines if x not in edge]
        chosen = edge + rest[: max(per_route - len(edge), 0)]
        chosen.sort(key=lambda x: x["ts"])
        for line in chosen:
            cases.append(
                {
                    "id": line["request_id"],
                    "route": route,
                    "history": [
                        {"role": h["role"], "content": strip_pii(h["content"])}
                        for h in line.get("history", [])
                    ],
                    "input": strip_pii(line["input"]),
                    "expect": expectation(line),
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("logs", help="JSONL production log")
    parser.add_argument("--per-route", type=int, default=100, help="cap per route (default 100)")
    args = parser.parse_args()

    with open(args.logs, encoding="utf-8") as handle:
        lines = [json.loads(row) for row in handle if row.strip()]
    for case in build(lines, args.per_route):
        sys.stdout.write(json.dumps(case, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
