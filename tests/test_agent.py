"""The live path's tool loop, driven by a scripted fake client. No network."""

from types import SimpleNamespace

import agent


class FakeClient:
    """Replays scripted responses and records what it was asked."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def text(content):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=content)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def tool_use(name, args, call_id="toolu_1"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name=name, input=args, id=call_id)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def test_tool_loop_runs_the_tool_and_returns_the_final_text():
    client = FakeClient(
        tool_use("get_schedule", {"date": "2026-09-08"}),
        text("Standup at 9:30 and the dentist at 14:00."),
    )
    case = {"route": "planner", "history": [], "input": "What's on my calendar today?"}

    result = agent.run_case("any-model", case, client=client)

    assert result["output"] == "Standup at 9:30 and the dentist at 14:00."
    assert result["tool_calls"] == [{"name": "get_schedule", "args": {"date": "2026-09-08"}}]
    assert result["input_tokens"] == 20 and result["output_tokens"] == 10
    second = client.requests[1]
    assert second["temperature"] == 0
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and "Team standup" in tool_result["content"]


def test_history_is_replayed_before_the_input():
    client = FakeClient(text('{"category": "other"}'))
    case = {
        "route": "triage",
        "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "input": "Dark mode please!",
    }
    agent.run_case("any-model", case, client=client)
    sent = client.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[-1]["content"] == "Dark mode please!"
    assert client.requests[0]["tools"] == []
