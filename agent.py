"""The app under test: three routes behind one `run_case` entry point.

This is the only file that talks to a model provider. `run_case(model, case)`
is the seam: swap the body for your own agent and the rest of the repo does
not care. It returns the shape the checks and the cache read:

    {"output": str, "tool_calls": [{"name": str, "args": dict}], "latency_ms": int,
     "input_tokens": int, "output_tokens": int}

The model names below are the "model config" the CI workflow triggers on.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from typing import Callable

BASELINE_MODEL = "claude-sonnet-5"  # what production runs today
CANDIDATE_MODEL = "claude-sonnet-5-5"  # what you want to move to

MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 5


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]


@dataclass(frozen=True)
class Route:
    system: str
    tools: tuple[Tool, ...] = ()


# --- planner: calendar, tasks, notes over fixed sample data -----------------

TODAY = dt.date(2026, 9, 8)  # frozen clock so answers are reproducible
EVENTS = [
    ("2026-09-08", "09:30", "10:00", "Team standup", "Zoom"),
    ("2026-09-08", "14:00", "15:00", "Dentist appointment", "Downtown Dental"),
    ("2026-09-09", "11:00", "12:00", "Design review with Maya", "Room 3B"),
    ("2026-09-10", "18:30", "20:00", "Yoga class", "Studio One"),
    ("2026-09-12", "10:00", "11:30", "Quarterly planning", "HQ"),
]
TASKS = [
    (1, "Renew passport", "2026-09-15", "pending"),
    (2, "Send invoice to Acme", "2026-09-09", "pending"),
    (3, "Buy birthday gift for Sam", "2026-09-11", "pending"),
    (4, "Book flights to Lisbon", "2026-09-02", "done"),
    (5, "Update résumé", "2026-08-28", "done"),
]
NOTES = [
    (1, "2026-08-20", "Dentist said to floss daily and come back in six months."),
    (
        2,
        "2026-08-30",
        "Lisbon trip ideas: Belém tower, pastel de nata at Manteigaria, day trip to Sintra.",
    ),
    (3, "2026-09-01", "Acme contract: net-30 payment terms, PO number 4471."),
    (4, "2026-09-05", "Gift ideas for Sam: vinyl records, hiking socks, coffee grinder."),
]

PLANNER_PROMPT = """You are Day Planner, a concise personal assistant. Today is Tuesday, 2026-09-08.
You have three tools: get_schedule (calendar events for one date), get_tasks (to-do items filtered by status) and search_notes (free-text search over saved notes).
Rules:
- Always call a tool before answering questions about events, tasks or notes; never invent data.
- Pick the single most specific tool for the question; call several only when the question genuinely spans several areas.
- Resolve relative dates ("tomorrow", "Friday", "next Monday") into YYYY-MM-DD yourself before calling get_schedule.
- Answer in at most three short sentences of plain text, no markdown, no bullet lists."""


def get_schedule(date: str) -> dict:
    try:
        iso = dt.date.fromisoformat(str(date).strip()).isoformat()
    except ValueError:
        return {"success": False, "error": f"Invalid date '{date}'; use YYYY-MM-DD."}
    events = [
        {"title": t, "start": s, "end": e, "location": loc}
        for day, s, e, t, loc in EVENTS
        if day == iso
    ]
    return {"success": True, "date": iso, "events": events}


def get_tasks(status: str = "pending") -> dict:
    value = str(status).strip().lower()
    if value not in ("pending", "done", "all"):
        return {"success": False, "error": f"Unknown status '{status}'; use pending, done or all."}
    rows = [r for r in TASKS if value == "all" or r[3] == value]
    return {
        "success": True,
        "status": value,
        "tasks": [{"id": i, "title": t, "due_date": d, "status": s} for i, t, d, s in rows],
    }


def search_notes(query: str) -> dict:
    words = [w for w in str(query).lower().split() if len(w) >= 3]
    hits = []
    for note_id, day, content in NOTES:
        score = sum(1 for w in words if w in content.lower())
        if score:
            hits.append((-score, note_id, {"id": note_id, "date": day, "content": content}))
    hits.sort(key=lambda h: (h[0], h[1]))
    return {"success": True, "query": query, "notes": [h[2] for h in hits]}


PLANNER_TOOLS = (
    Tool(
        "get_schedule",
        "Get the user's calendar events for a single date.",
        {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "The date in YYYY-MM-DD format"}
            },
            "required": ["date"],
        },
        get_schedule,
    ),
    Tool(
        "get_tasks",
        "List the user's to-do tasks, optionally filtered by status.",
        {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["pending", "done", "all"]}},
            "required": [],
        },
        get_tasks,
    ),
    Tool(
        "search_notes",
        "Search the user's saved free-text notes by keyword.",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords to search for"}},
            "required": ["query"],
        },
        search_notes,
    ),
)

# --- converter: units, currency, dates --------------------------------------

CONVERTER_PROMPT = """You are Converter, a precise unit, currency and date calculator.
Never do arithmetic yourself: always call convert_units, convert_currency or days_between and report the tool's result.
Normalise the user's wording into tool arguments: the numeric value as a plain number (no thousands separators), units as short codes (km, mi, m, ft, kg, lb, c, f, l, gal), currencies as ISO codes (USD, EUR, GBP, JPY, CHF), dates as YYYY-MM-DD (assume the current year, 2026, when the year is missing).
If a request needs several conversions, call the tool once per conversion.
Reply in one or two plain-text sentences with the results rounded sensibly; if a tool returns an error, explain it briefly."""

LENGTH = {"km": 1000.0, "mi": 1609.344, "m": 1.0, "ft": 0.3048}
MASS = {"kg": 1.0, "lb": 0.45359237}
VOLUME = {"l": 1.0, "gal": 3.785411784}
RATES_PER_USD = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.0, "CHF": 0.88}


def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    src, dst = str(from_unit).lower(), str(to_unit).lower()
    for table in (LENGTH, MASS, VOLUME):
        if src in table and dst in table:
            return {"success": True, "result": round(float(value) * table[src] / table[dst], 3)}
    if {src, dst} == {"c", "f"}:
        result = value * 9 / 5 + 32 if src == "c" else (value - 32) * 5 / 9
        return {"success": True, "result": round(result, 3)}
    return {"success": False, "error": f"Cannot convert from '{from_unit}' to '{to_unit}'."}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    src, dst = str(from_currency).upper(), str(to_currency).upper()
    if src not in RATES_PER_USD or dst not in RATES_PER_USD:
        return {"success": False, "error": "Unsupported currency; use USD, EUR, GBP, JPY or CHF."}
    return {
        "success": True,
        "result": round(float(amount) / RATES_PER_USD[src] * RATES_PER_USD[dst], 2),
    }


def days_between(start_date: str, end_date: str) -> dict:
    try:
        start, end = (dt.date.fromisoformat(str(d).strip()) for d in (start_date, end_date))
    except ValueError:
        return {"success": False, "error": "Invalid date; use YYYY-MM-DD."}
    return {"success": True, "days": (end - start).days}


CONVERTER_TOOLS = (
    Tool(
        "convert_units",
        "Convert a value between units of length (km, mi, m, ft), mass (kg, lb), temperature (c, f) or volume (l, gal).",
        {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        convert_units,
    ),
    Tool(
        "convert_currency",
        "Convert an amount between currencies (USD, EUR, GBP, JPY, CHF) at fixed sample rates.",
        {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "ISO 4217 code"},
                "to_currency": {"type": "string", "description": "ISO 4217 code"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
        convert_currency,
    ),
    Tool(
        "days_between",
        "Count the days from one calendar date to another.",
        {
            "type": "object",
            "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}},
            "required": ["start_date", "end_date"],
        },
        days_between,
    ),
)

# --- triage: structured JSON, no tools --------------------------------------

TRIAGE_PROMPT = """You are a support-ticket triage classifier. Read the customer's message and return ONLY a JSON object — no markdown fences, no commentary — with exactly these keys:
- "category": one of "billing", "bug", "feature_request", "account", "other"
- "priority": one of "low", "medium", "high", "urgent" (urgent means the customer is blocked or losing money right now)
- "sentiment": one of "negative", "neutral", "positive"
- "summary": one sentence of at most 20 words describing the issue
- "suggested_reply": a polite 2-3 sentence customer-facing reply that acknowledges the issue and states the next step"""

ROUTES = {
    "planner": Route(PLANNER_PROMPT, PLANNER_TOOLS),
    "converter": Route(CONVERTER_PROMPT, CONVERTER_TOOLS),
    "triage": Route(TRIAGE_PROMPT),
}


# --- the seam ---------------------------------------------------------------


def run_case(model: str, case: dict, client=None) -> dict:
    """One golden case through the agent on `model`. Temperature 0, bounded tool loop."""
    import anthropic

    client = client or anthropic.Anthropic()
    route = ROUTES[case["route"]]
    tools = [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in route.tools
    ]
    tool_map = {t.name: t for t in route.tools}
    messages = list(case.get("history", [])) + [{"role": "user", "content": case["input"]}]

    calls: list[dict] = []
    output = ""
    input_tokens = output_tokens = 0
    started = time.perf_counter()
    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=route.system,
            tools=tools,
            messages=list(messages),
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        requested = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})
        if not requested:
            output = "".join(b.text for b in response.content if b.type == "text")
            break
        results = []
        for block in requested:
            args = dict(block.input) if isinstance(block.input, dict) else {}
            calls.append({"name": block.name, "args": args})
            tool = tool_map.get(block.name)
            result = tool.fn(**args) if tool else {"success": False, "error": "unknown tool"}
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
            )
        messages.append({"role": "user", "content": results})

    return {
        "output": output,
        "tool_calls": calls,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
