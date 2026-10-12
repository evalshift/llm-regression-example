"""Run the golden set against a baseline and a candidate model, diff the pass
rates per check, and fail if any check dropped more than --max-drop.

    python run_eval.py                                  # models from agent.py
    python run_eval.py --candidate claude-opus-5-5      # audition another model
    python run_eval.py --repeats 3                      # 3 samples per case

Model outputs are cached per model in results/<model>.jsonl. A model with a
complete cache never gets called again, which is what lets CI reuse the
baseline and what lets this repo run offline. Delete the file to re-run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import agent
from checks import run_checks

RESULTS_DIR = Path("results")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(row) for row in handle if row.strip()]


def results_for(model: str, cases: list[dict], repeats: int) -> dict[str, list[dict]]:
    """Recorded samples per case id, calling the model for whatever is missing."""
    path = RESULTS_DIR / f"{model}.jsonl"
    samples: dict[str, list[dict]] = defaultdict(list)
    for row in load_jsonl(path):
        samples[row["id"]].append(row)

    missing = [(case, repeats - len(samples[case["id"]])) for case in cases]
    missing = [(case, n) for case, n in missing if n > 0]
    if missing:
        total = sum(n for _, n in missing)
        print(f"{model}: calling the model for {total} samples", file=sys.stderr)
        RESULTS_DIR.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for case, n in missing:
                for _ in range(n):
                    row = {"id": case["id"], "route": case["route"], "model": model}
                    row.update(agent.run_case(model, case))
                    samples[case["id"]].append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return samples


def score(cases: list[dict], samples: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """case id -> check name -> fraction of samples that passed."""
    scores: dict[str, dict[str, float]] = {}
    for case in cases:
        per_check: dict[str, list[bool]] = defaultdict(list)
        for row in samples[case["id"]]:
            for check in run_checks(case, row):
                per_check[check.name].append(check.passed)
        scores[case["id"]] = {name: statistics.fmean(v) for name, v in per_check.items()}
    return scores


def pass_rates(cases: list[dict], scores: dict) -> dict[tuple[str, str], tuple[float, int]]:
    """(route, check) -> (pass rate across cases, n cases)."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case in cases:
        for name, value in scores[case["id"]].items():
            buckets[(case["route"], name)].append(value)
    return {key: (statistics.fmean(v), len(v)) for key, v in buckets.items()}


def describe(row: dict) -> str:
    calls = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in row["tool_calls"])
    text = row["output"].strip().replace("\n", " ")
    if len(text) > 110:
        text = text[:107] + "..."
    if calls and text:
        return f"{calls}  then  {text!r}"
    return calls or (repr(text) if text else "(no text, no tool call)")


def print_report(
    cases: list[dict],
    baseline: str,
    candidate: str,
    base_scores: dict,
    cand_scores: dict,
    base_samples: dict,
    cand_samples: dict,
    max_drop: float,
) -> list[tuple[str, str, float]]:
    base_rates = pass_rates(cases, base_scores)
    cand_rates = pass_rates(cases, cand_scores)

    print(f"{'route':<10} {'check':<15} {'n':>3} {'baseline':>9} {'candidate':>10} {'delta':>7}")
    drops = []
    for key in sorted(base_rates):
        base, n = base_rates[key]
        cand = cand_rates.get(key, (0.0, 0))[0]
        delta = cand - base
        flag = ""
        if delta < -max_drop:
            flag = "  REGRESSED"
            drops.append((key[0], key[1], delta))
        elif delta > max_drop:
            flag = "  improved"
        print(
            f"{key[0]:<10} {key[1]:<15} {n:>3} {base:>8.1%} {cand:>9.1%} {delta * 100:>+6.1f}{flag}"
        )

    print()
    print(f"Cases where {candidate} fails a check {baseline} passed:")
    print()
    changed = 0
    for case in cases:
        worse = [
            (name, base_scores[case["id"]].get(name, 0.0), value)
            for name, value in cand_scores[case["id"]].items()
            if value < base_scores[case["id"]].get(name, 0.0)
        ]
        if not worse:
            continue
        changed += 1
        print(f"{case['id']}  {case['input']!r}")
        cand_row = cand_samples[case["id"]][0]
        details = {c.name: c.detail for c in run_checks(case, cand_row)}
        for name, _, _ in worse:
            print(f"  {name:<14} {details.get(name, '')}")
        print(f"  {baseline + ':':<{len(candidate) + 1}}  {describe(base_samples[case['id']][0])}")
        print(f"  {candidate + ':':<{len(candidate) + 1}}  {describe(cand_row)}")
        print()
    if not changed:
        print("  none")
        print()

    print_economics(cases, baseline, candidate, base_samples, cand_samples)
    return drops


def print_economics(cases, baseline, candidate, base_samples, cand_samples) -> None:
    rows = []
    for route in sorted({c["route"] for c in cases}):
        ids = [c["id"] for c in cases if c["route"] == route]
        for field, fmt in (("latency_ms", "{:.0f} ms"), ("cost_usd", "${:.4f}")):
            base = [r[field] for i in ids for r in base_samples[i] if field in r]
            cand = [r[field] for i in ids for r in cand_samples[i] if field in r]
            if not base or not cand:
                continue
            b, c = statistics.fmean(base), statistics.fmean(cand)
            rows.append((route, field, fmt.format(b), fmt.format(c), (c - b) / b))
    if not rows:
        return
    print(f"{'route':<10} {'metric':<11} {baseline:>16} {candidate:>16} {'change':>8}")
    for route, field, b, c, change in rows:
        print(f"{route:<10} {field:<11} {b:>16} {c:>16} {change:>+7.0%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--golden", default="golden.jsonl")
    parser.add_argument("--baseline", default=agent.BASELINE_MODEL)
    parser.add_argument("--candidate", default=agent.CANDIDATE_MODEL)
    parser.add_argument("--repeats", type=int, default=1, help="samples per case (default 1)")
    parser.add_argument(
        "--max-drop",
        type=float,
        default=0.02,
        help="fail if any (route, check) pass rate drops by more than this fraction (default 0.02)",
    )
    args = parser.parse_args()

    cases = load_jsonl(Path(args.golden))
    if not cases:
        print(f"no cases in {args.golden}", file=sys.stderr)
        return 2
    base_samples = results_for(args.baseline, cases, args.repeats)
    cand_samples = results_for(args.candidate, cases, args.repeats)
    base_scores = score(cases, base_samples)
    cand_scores = score(cases, cand_samples)

    print(f"{len(cases)} cases, {args.repeats} sample(s) each: {args.baseline} -> {args.candidate}")
    print()
    drops = print_report(
        cases,
        args.baseline,
        args.candidate,
        base_scores,
        cand_scores,
        base_samples,
        cand_samples,
        args.max_drop,
    )
    print()
    if drops:
        worst = min(drops, key=lambda d: d[2])
        print(
            f"FAIL: {len(drops)} check(s) dropped more than {args.max_drop:.0%}"
            f" (worst: {worst[0]}/{worst[1]} {worst[2] * 100:+.1f} points)"
        )
        return 1
    print(f"PASS: no check dropped more than {args.max_drop:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
