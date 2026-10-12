from build_golden import build, strip_pii


def line(request_id, route, text, ts, **extra):
    base = {
        "request_id": request_id,
        "ts": ts,
        "route": route,
        "history": [],
        "input": text,
        "output": "",
        "tool_calls": [],
    }
    base.update(extra)
    return base


def test_strips_emails_and_phone_numbers():
    assert (
        strip_pii("mail me at jane.doe+x@example.com or call +1 (555) 123-4567")
        == "mail me at [email] or call [phone]"
    )


def test_drops_repeated_inputs_and_keeps_oldest():
    lines = [
        line("r2", "converter", "Convert 5 kg to pounds.", "2026-10-12T10:00:01Z"),
        line("r1", "converter", "convert 5 KG to pounds. ", "2026-10-12T10:00:00Z"),
    ]
    assert [c["id"] for c in build(lines, per_route=10)] == ["r1"]


def test_edge_tagged_lines_survive_the_per_route_cap():
    lines = [
        line("r1", "converter", "5 kg to lb", "2026-10-12T10:00:00Z"),
        line("r2", "converter", "3 mi to km", "2026-10-12T10:00:01Z"),
        line("r3", "converter", "10 km to kg", "2026-10-12T10:00:02Z", tags=["edge"]),
    ]
    assert [c["id"] for c in build(lines, per_route=2)] == ["r1", "r3"]


def test_expectations_come_from_what_production_did():
    tool_line = line(
        "p1",
        "planner",
        "What's on today?",
        "2026-10-12T10:00:00Z",
        tool_calls=[{"name": "get_schedule", "args": {"date": "2026-09-08"}}],
    )
    json_line = line(
        "t1",
        "triage",
        "Charged twice, refund me.",
        "2026-10-12T10:00:01Z",
        output='```json\n{"category": "billing", "priority": "urgent", "sentiment": "negative",'
        ' "summary": "s", "suggested_reply": "r"}\n```',
    )
    cases = {c["id"]: c for c in build([tool_line, json_line], per_route=10)}
    assert cases["p1"]["expect"] == {
        "tools": [{"name": "get_schedule", "args": {"date": "2026-09-08"}}]
    }
    assert cases["t1"]["expect"] == {
        "json": {"category": "billing", "priority": "urgent", "sentiment": "negative"}
    }
