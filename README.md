# Catch LLM regressions before a model upgrade ships

The runnable example behind
[How to catch LLM regressions before a model upgrade ships](https://www.evalshift.dev/blog/llm-regression-testing-model-upgrade).
Production logs become a golden set, deterministic assertions score it, a
baseline-vs-candidate diff shows what changed, and a GitHub Actions job fails
the pull request when a check drops.

The data is real: 30 recorded turns from a small three-route agent (a day
planner and a unit converter with tools, a ticket triage classifier that
returns JSON), replayed on `claude-sonnet-5` and `claude-sonnet-5-5`. The
recorded outputs are committed under `results/`, so the diff runs offline
with no API key.

```bash
uv sync                                   # or: pip install -e . && pip install pytest
uv run python build_golden.py logs/production.jsonl > golden.jsonl
uv run python run_eval.py                 # offline: reads results/*.jsonl
```

The last command prints a pass-rate table per route and check, every case
where the candidate fails a check the baseline passed, the latency and cost
shift, and exits 1:

```
route      check             n  baseline  candidate   delta
converter  answered          1   100.0%      0.0% -100.0  REGRESSED
converter  tool_set         10   100.0%     90.0%  -10.0  REGRESSED
planner    arg_format       11   100.0%     90.9%   -9.1  REGRESSED
planner    tool_set         11   100.0%     81.8%  -18.2  REGRESSED
triage     fields            9    88.9%     66.7%  -22.2  REGRESSED
triage     json_strict       9    11.1%    100.0%  +88.9  improved
...
planner-000  "What's on my calendar today?"
  tool_args      get_schedule.date: '2026-09-08' -> 'today'
  arg_format     get_schedule.date='today'
  claude-sonnet-5:    get_schedule({"date": "2026-09-08"})
  claude-sonnet-5-5:  get_schedule({"date": "today"})
```

## Layout

| File | What it does |
| --- | --- |
| `logs/production.jsonl` | One line per production turn: route, history, input, output, tool calls. The input to everything else. |
| `build_golden.py` | Logs to `golden.jsonl`: strips PII, drops repeats, samples per route, keeps `edge`-tagged lines, derives an `expect` block from what production did. |
| `checks.py` | The assertions. Baseline checks (`fields`, `tool_set`, `tool_args`) compare against production; contract checks (`json_strict`, `json_schema`, `arg_format`, `summary_length`) encode what the prompt promises. |
| `run_eval.py` | Runs baseline and candidate over the golden set (cached per model in `results/`), scores every sample, prints the diff, exits 1 on a drop larger than `--max-drop`. |
| `agent.py` | The app under test: prompts, tools and model names, behind one `run_case()` seam. |
| `.github/workflows/llm-regression.yml` | The gate: triggers on prompt, model, assertion or golden-set changes, caches the baseline, fails the PR. |
| `evalshift.yaml` | The same flow with the EvalShift CLI instead of the three scripts. |

## Run it live

Delete a model's file under `results/` and `run_eval.py` calls that model
again; `--repeats 3` takes three samples per case and reports pass fractions
instead of single outcomes. Needs `ANTHROPIC_API_KEY`.

```bash
rm results/claude-sonnet-5-5.jsonl
uv run python run_eval.py --repeats 3
uv run python run_eval.py --candidate claude-opus-5-5     # audition another model
```

## Use it on your own app

1. Replace the body of `run_case()` in `agent.py` with a call into your agent.
   It has to return `output`, `tool_calls`, `latency_ms` and the token counts.
2. Write your production turns as lines of `logs/production.jsonl`. Tag the
   ones you know are edge cases.
3. Edit the tables at the top of `checks.py`: the JSON fields and enums your
   output promises, the argument shapes your tools require, the arguments the
   model is allowed to phrase freely.
4. Commit `golden.jsonl`, add `results/` to `.gitignore`, add the workflow,
   set `ANTHROPIC_API_KEY` as a repository secret.

## The shortcut

[EvalShift](https://www.evalshift.dev) does the same thing with a capture SDK
instead of a log exporter, a YAML file instead of `checks.py`, and paired
statistics plus a policy verdict instead of a pass-rate threshold. The
`evalshift.yaml` here is the equivalent of this repo's assertions:

```bash
pip install evalshift
evalshift capture sync      # recorded runs -> golden suite
evalshift compare           # baseline vs candidate -> report.html + verdict
```

## Tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Apache 2.0.
