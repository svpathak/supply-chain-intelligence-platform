"""Lightweight Groq chat-completions tool-calling loop and audit logging."""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.assistant.tools import AssistantToolError, AssistantTools

logger = logging.getLogger("supply_chain.genai")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_shipments",
            "description": (
                "Read shipments from the curated data product. Use date_from/date_to for a time window. "
                "Set aggregate_by_route=true to return routes ranked by average delay for that booking-date window. "
                "Otherwise returns shipment rows."
            ),
            "parameters": {
                "type": "object", "properties": {
                    "filters": {"type": "object", "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "status": {"type": "string"},
                        "date_from": {"type": "string", "description": "Inclusive YYYY-MM-DD"},
                        "date_to": {"type": "string", "description": "Inclusive YYYY-MM-DD"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                        "offset": {"type": "integer", "minimum": 0},
                        "aggregate_by_route": {"type": "boolean"},
                    }, "additionalProperties": False}
                }, "required": ["filters"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_route_stats",
            "description": "Get historical average delay hours, on-time rate and shipment count for one origin/destination port pair.",
            "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"}, "destination": {"type": "string"}
            }, "required": ["origin", "destination"], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_delay",
            "description": "Call the saved ML model through POST /predict-delay for a shipment ID.",
            "parameters": {"type": "object", "properties": {
                "shipment_id": {"type": "string"}
            }, "required": ["shipment_id"], "additionalProperties": False},
        },
    },
]

DOMAIN_TERMS = re.compile(
    r"\b(shipments?|routes?|ports?|delays?|late|on[- ]?time|congestion|vessels?|"
    r"bookings?|arrivals?|departures?|containers?|cargo|shipping|supply chain|delay[- ]risk)\b|\bSHP[-_]\w+",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are the Supply Chain Intelligence assistant for this shipment dataset.
Answer only questions about shipments, routes, ports, shipping delays, on-time performance,
congestion, cargo, and the supplied platform's delay-risk predictions. For questions requiring
current or historical dataset facts, call the appropriate tool; never invent rows, metrics, or
predictions. Use query_shipments with aggregate_by_route=true for ranked route questions.
For 'this quarter', derive the quarter start and end dates from the current date supplied below.
Use get_route_stats for one named route and predict_delay for one shipment ID.
When summarising a prediction, state that the model is experimental and not approved for
operational decisions. Cite factual claims with the exact tool name in square brackets, e.g.
[tool: get_route_stats]. Do not claim a tool was used if it was not called.
Current date: {current_date}.
"""


class GroqSupplyChainAssistant:
    def __init__(
        self,
        tools: AssistantTools | None = None,
        client: Any | None = None,
        model: str | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.tools = tools or AssistantTools()
        self.client = client
        self.model = model or settings.groq_model
        self.audit_path = Path(audit_path or settings.genai_audit_log)
        self.used_tools: list[str] = []

    def _client(self):
        if self.client is None:
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or the environment.")
            from groq import Groq
            self.client = Groq(api_key=settings.groq_api_key)
        return self.client

    def _audit(self, event: str, **payload: Any) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        logger.info("genai_%s", event, extra={"path": str(self.audit_path)})

    def answer(self, question: str) -> str:
        question = question.strip()
        self.used_tools = []
        if not question:
            return "Please enter a supply-chain question."
        # Explicit guardrail 1: deterministic domain gate before any LLM request.
        if not DOMAIN_TERMS.search(question):
            refusal = "I can only help with questions about shipments, routes, ports, shipping delays, on-time performance and delay risk."
            self._audit("off_domain_refusal", user_question=question, response=refusal)
            return refusal

        system_message = SYSTEM_PROMPT.format(current_date=date.today().isoformat())
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ]
        client = self._client()

        while True:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=900,
            )
            message = completion.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            response_snapshot = {
                "content": getattr(message, "content", None),
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                    for call in tool_calls
                ],
            }
            usage = getattr(completion, "usage", None)
            self._audit(
                "llm_call",
                model=self.model,
                prompt_messages=messages,
                tool_definitions=[item["function"]["name"] for item in TOOLS],
                response=response_snapshot,
                usage=(usage.model_dump() if hasattr(usage, "model_dump") else str(usage) if usage else None),
            )

            if not tool_calls:
                answer = (getattr(message, "content", None) or "I could not produce an answer from the available data.").strip()
                # Explicit guardrail 2: deterministic provenance footer for factual answers.
                if self.used_tools:
                    sources = ", ".join(dict.fromkeys(self.used_tools))
                    answer = f"{answer}\n\nSources: {sources}."
                else:
                    answer = f"{answer}\n\nNo data tools were called; no shipment-specific facts were retrieved."
                self._audit("final_answer", user_question=question, tools_used=list(dict.fromkeys(self.used_tools)), output=answer)
                return answer

            assistant_message = {
                "role": "assistant",
                "content": getattr(message, "content", None),
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in tool_calls
                ],
            }
            messages.append(assistant_message)
            for call in tool_calls:
                name = call.function.name
                arguments: dict[str, Any] = {}
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise AssistantToolError("Tool arguments must be a JSON object")
                    result = self.tools.execute(name, arguments)
                    self.used_tools.append(name)
                except Exception as exc:
                    # Return a bounded tool error to the model; never expose a traceback.
                    result = {"error": str(exc) or "Tool execution failed."}
                self._audit("tool_execution", tool=name, arguments=arguments, response=result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": json.dumps(result, default=str, ensure_ascii=False),
                })
