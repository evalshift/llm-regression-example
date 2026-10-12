"""The assertions, exercised on the shapes of failure the post describes.

Every case here mirrors something that actually happened in the recorded
Sonnet 5 -> Sonnet 5.5 run (see results/), reduced to the smallest input that
shows it.
"""

from checks import run_checks


def by_name(results):
    return {r.name: r for r in results}


# --- JSON routes (triage) ----------------------------------------------------

TRIAGE_CASE = {
    "id": "triage-000",
    "route": "triage",
    "expect": {"json": {"category": "billing", "priority": "urgent", "sentiment": "negative"}},
}

BARE = (
    '{"category": "billing", "priority": "urgent", "sentiment": "negative", '
    '"summary": "Customer was double-charged and wants a refund today.", '
    '"suggested_reply": "Sorry about that, refunding now."}'
)


def test_bare_json_passes_every_json_check():
    checks = by_name(run_checks(TRIAGE_CASE, {"output": BARE, "tool_calls": []}))
    assert {n for n, r in checks.items() if not r.passed} == set()


def test_fenced_json_fails_only_the_strict_parse():
    fenced = "```json\n" + BARE + "\n```"
    checks = by_name(run_checks(TRIAGE_CASE, {"output": fenced, "tool_calls": []}))
    assert not checks["json_strict"].passed
    assert checks["json_schema"].passed
    assert checks["fields"].passed


def test_schema_valid_but_downgraded_priority_fails_fields():
    out = BARE.replace('"priority": "urgent"', '"priority": "high"')
    checks = by_name(run_checks(TRIAGE_CASE, {"output": out, "tool_calls": []}))
    assert checks["json_schema"].passed
    assert not checks["fields"].passed
    assert "priority" in checks["fields"].detail


def test_unknown_enum_value_fails_schema():
    out = BARE.replace('"priority": "urgent"', '"priority": "p0"')
    checks = by_name(run_checks(TRIAGE_CASE, {"output": out, "tool_calls": []}))
    assert not checks["json_schema"].passed


def test_summary_over_twenty_words_fails_length():
    long = "word " * 25
    out = BARE.replace("Customer was double-charged and wants a refund today.", long.strip())
    checks = by_name(run_checks(TRIAGE_CASE, {"output": out, "tool_calls": []}))
    assert not checks["summary_length"].passed


# --- tool routes (planner, converter) ----------------------------------------

PLANNER_CASE = {
    "id": "planner-007",
    "route": "planner",
    "expect": {"tools": [{"name": "get_schedule", "args": {"date": "2026-09-08"}}]},
}


def test_same_call_passes_everything():
    calls = [{"name": "get_schedule", "args": {"date": "2026-09-08"}}]
    checks = by_name(run_checks(PLANNER_CASE, {"output": "", "tool_calls": calls}))
    assert all(r.passed for r in checks.values())


def test_relative_date_fails_the_argument_contract():
    calls = [{"name": "get_schedule", "args": {"date": "today"}}]
    checks = by_name(run_checks(PLANNER_CASE, {"output": "", "tool_calls": calls}))
    assert checks["tool_set"].passed
    assert not checks["arg_format"].passed
    assert not checks["tool_args"].passed


def test_extra_tool_call_fails_tool_set():
    calls = [
        {"name": "get_schedule", "args": {"date": "2026-09-08"}},
        {"name": "get_tasks", "args": {"status": "pending"}},
    ]
    checks = by_name(run_checks(PLANNER_CASE, {"output": "", "tool_calls": calls}))
    assert not checks["tool_set"].passed
    assert "get_tasks" in checks["tool_set"].detail


def test_free_text_argument_drift_is_reported_not_failed():
    case = {
        "id": "planner-009",
        "route": "planner",
        "expect": {
            "tools": [{"name": "search_notes", "args": {"query": "Acme PO number contract"}}]
        },
    }
    calls = [{"name": "search_notes", "args": {"query": "Acme PO number"}}]
    checks = by_name(run_checks(case, {"output": "", "tool_calls": calls}))
    assert checks["tool_args"].passed
    assert "query" in checks["tool_args"].detail


def test_answering_without_a_tool_where_none_is_expected_passes():
    case = {"id": "converter-008", "route": "converter", "expect": {"tools": []}}
    result = {"output": "Kilometres measure distance, kilograms mass.", "tool_calls": []}
    checks = by_name(run_checks(case, result))
    assert checks["tool_set"].passed
    assert checks["answered"].passed


def test_hallucinated_tool_call_fails_tool_set_and_answered():
    case = {"id": "converter-008", "route": "converter", "expect": {"tools": []}}
    calls = [{"name": "convert_units", "args": {"value": 10, "from_unit": "km", "to_unit": "kg"}}]
    checks = by_name(run_checks(case, {"output": "", "tool_calls": calls}))
    assert not checks["tool_set"].passed
    assert not checks["answered"].passed


def test_non_iso_currency_code_fails_the_argument_contract():
    case = {
        "id": "converter-001",
        "route": "converter",
        "expect": {
            "tools": [
                {
                    "name": "convert_currency",
                    "args": {"amount": 120, "from_currency": "EUR", "to_currency": "JPY"},
                }
            ]
        },
    }
    calls = [
        {
            "name": "convert_currency",
            "args": {"amount": 120, "from_currency": "euros", "to_currency": "JPY"},
        }
    ]
    checks = by_name(run_checks(case, {"output": "", "tool_calls": calls}))
    assert not checks["arg_format"].passed
