import json
from types import SimpleNamespace

from app.assistant.groq_assistant import GroqSupplyChainAssistant


class FakeTools:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return {"shipment_count": 17, "on_time_rate": 0.82}


class FakeGroqClient:
    def __init__(self):
        self.requests = []
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="get_route_stats",
                arguments=json.dumps({"origin": "Shanghai", "destination": "Rotterdam"}),
            ),
        )
        first_message = SimpleNamespace(content=None, tool_calls=[tool_call])
        final_message = SimpleNamespace(
            content="The route has an 82% on-time rate across 17 shipments.",
            tool_calls=None,
        )
        self.responses = [first_message, final_message]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 123}),
        )


def test_off_domain_question_is_refused_without_llm_call(tmp_path):
    client = FakeGroqClient()
    assistant = GroqSupplyChainAssistant(client=client, audit_path=tmp_path / "audit.jsonl")
    answer = assistant.answer("Can you write me a poem about Mars?")
    assert "only help with questions about" in answer
    assert client.requests == []


def test_tool_call_loop_executes_tool_cites_source_and_audits(tmp_path):
    client = FakeGroqClient()
    tools = FakeTools()
    audit = tmp_path / "audit.jsonl"
    assistant = GroqSupplyChainAssistant(tools=tools, client=client, audit_path=audit)
    answer = assistant.answer("What is the on-time rate for the Shanghai to Rotterdam route?")
    assert "82%" in answer
    assert "Sources: get_route_stats." in answer
    assert tools.calls == [("get_route_stats", {"origin": "Shanghai", "destination": "Rotterdam"})]
    assert len(client.requests) == 2
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [event["event"] for event in events].count("llm_call") == 2
    assert any(event["event"] == "tool_execution" for event in events)
    assert any(event["event"] == "final_answer" for event in events)
    assert events[0]["prompt_messages"][1]["content"].startswith("What is the on-time")
