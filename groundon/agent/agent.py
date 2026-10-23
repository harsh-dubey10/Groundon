"""Agentic RAG: the LLM decides which tools to call, we execute them, and loop until it answers.

    user question
        │
        ▼
    LLM (with tool schemas) ──► wants tool? ──yes──► run tool ──► append result ──┐
        ▲                          │ no                                           │
        └──────────────────────────┼──────────────────────────────────────────────┘
                                   ▼
                              final answer (+ citation / groundedness checks)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from groundon.agent.tools import CurrencyTool, DocumentSearchTool, SqlTool, ToolError, tool_schemas
from groundon.generation.grounding import (CitationReport, Verification, check_citations, normalize_citations,
                                           verify_groundedness)
from groundon.generation.llm import LLMClient
from groundon.generation.prompts import AGENT_SYSTEM
from groundon.generation.rag import Source
from groundon.retrieval.retriever import Retriever

MAX_TOOL_RESULT_CHARS = 6000


@dataclass
class ToolStep:
    tool: str
    arguments: dict
    result: str
    error: bool
    duration_ms: int


@dataclass
class AgentAnswer:
    answer: str
    steps: list[ToolStep]
    sources: list[Source]
    citations: CitationReport | None
    verification: Verification | None
    stopped_reason: str  # "answered" | "max_steps"
    timings_ms: dict = field(default_factory=dict)


class Agent:
    def __init__(self, llm: LLMClient, retriever: Retriever, sql_tool: SqlTool, currency_tool: CurrencyTool,
                 *, max_steps: int = 6, verifier_model: str,
                 is_confident: Callable[[float, str], bool] | None = None):
        self.llm = llm
        self.retriever = retriever
        self.sql_tool = sql_tool
        self.currency_tool = currency_tool
        self.max_steps = max_steps
        self.verifier_model = verifier_model
        self.is_confident = is_confident
        self._schemas = tool_schemas(sql_tool.schema())

    def run(self, question: str, *, verify: bool = False) -> AgentAnswer:
        t0 = time.perf_counter()
        doc_search = DocumentSearchTool(self.retriever, is_confident=self.is_confident)  # fresh per run: holds this run's sources
        tools = {
            "search_documents": lambda a: doc_search(a["query"]),
            "query_database": lambda a: self.sql_tool(a["sql"]),
            "convert_currency": lambda a: self.currency_tool(float(a["amount"]), a["from_currency"], a["to_currency"]),
        }

        messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}]
        steps: list[ToolStep] = []
        evidence: list[str] = []  # tool results exactly as the model saw them, for the verifier
        answer, stopped = "", "max_steps"

        for _ in range(self.max_steps):
            result = self.llm.chat(messages, tools=self._schemas)
            if not result.tool_calls:
                answer, stopped = result.content.strip(), "answered"
                break

            messages.append({
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in result.tool_calls
                ],
            })
            for call in result.tool_calls:
                ts = time.perf_counter()
                output, is_error = self._execute(tools, call.name, call.arguments)
                steps.append(ToolStep(call.name, call.arguments, output[:1500], is_error,
                                      round((time.perf_counter() - ts) * 1000)))
                seen = output[:MAX_TOOL_RESULT_CHARS]
                evidence.append(f"[{call.name}({json.dumps(call.arguments)})]\n{seen}")
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": seen})
        else:
            # Out of steps: ask for a best-effort answer from what was gathered, without tools.
            messages.append({"role": "user", "content": "Stop calling tools. Answer now using only the tool results above."})
            answer = self.llm.chat(messages).content.strip()

        answer = normalize_citations(answer)
        citations = check_citations(answer, {s.label for s in doc_search.sources}) if doc_search.sources else None
        verification = None
        if verify:
            verification = verify_groundedness(self.llm, self.verifier_model,
                                               "\n\n".join(evidence) or "(no tool results)", answer)

        return AgentAnswer(answer=answer, steps=steps, sources=doc_search.sources, citations=citations,
                           verification=verification, stopped_reason=stopped,
                           timings_ms={"total": round((time.perf_counter() - t0) * 1000)})

    @staticmethod
    def _execute(tools: dict, name: str, arguments: dict) -> tuple[str, bool]:
        if name not in tools:
            return f"Error: unknown tool '{name}'. Available: {list(tools)}", True
        try:
            return tools[name](arguments), False
        except KeyError as e:
            return f"Error: missing argument {e}", True
        except (ToolError, ValueError, TypeError) as e:
            return f"Error: {e}", True
