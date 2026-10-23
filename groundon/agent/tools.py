"""Tools the agent can call. Each tool = a JSON schema the LLM sees + a Python function we run.

The LLM never executes anything itself: it only *requests* a call (name + JSON args);
our code validates the args, runs the function, and feeds the result back.
"""

import json
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from groundon.generation.rag import Source, format_documents, to_sources
from groundon.retrieval.retriever import Retriever

MAX_SQL_ROWS = 50


class ToolError(Exception):
    """An error the agent should see (so it can correct itself)."""


# ---------------- search_documents ----------------

@dataclass
class DocumentSearchTool:
    retriever: Retriever
    top_k: int = 4
    # The same abstain gate as RAG mode: (confidence, confidence_type) -> good enough to show the model?
    is_confident: Callable[[float, str], bool] | None = None
    sources: list[Source] = field(default_factory=list)  # accumulated across calls for citations

    def __call__(self, query: str) -> str:
        retrieval = self.retriever.retrieve(query, top_k=self.top_k)
        if retrieval.chunks and self.is_confident and not self.is_confident(retrieval.confidence, retrieval.confidence_type):
            return (f"No relevant documents found (best {retrieval.confidence_type} score "
                    f"{retrieval.confidence:.3f} below threshold).")
        new = to_sources(retrieval.chunks, start=len(self.sources) + 1)  # S-labels unique per agent run
        self.sources.extend(new)
        if not new:
            return "No documents found."
        return format_documents(new)


# ---------------- query_database ----------------

_READ_ONLY_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


@dataclass
class SqlTool:
    db_path: Path
    timeout_s: float = 5.0

    def schema(self) -> str:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT sql FROM sqlite_master WHERE type = 'table'").fetchall()
        return "\n".join(r[0] for r in rows)

    def _connect(self) -> sqlite3.Connection:
        # Layer 1: the connection itself is read-only at the SQLite level.
        db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        # Layer 2: an authorizer that denies anything except reads.
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
        db.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        return db

    def __call__(self, sql: str) -> str:
        sql = sql.strip().rstrip(";")
        # Layer 3: cheap syntactic checks for a clearer error message.
        if not _READ_ONLY_START.match(sql) or ";" in sql:
            raise ToolError("Only a single read-only SELECT statement is allowed.")
        # Layer 4: a time limit, since read-only queries can still run for ages (huge cross joins).
        deadline = time.monotonic() + self.timeout_s
        try:
            with closing(self._connect()) as db:
                db.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
                cursor = db.execute(sql)
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
        except sqlite3.OperationalError as e:
            if time.monotonic() > deadline:
                raise ToolError(f"Query stopped: exceeded the {self.timeout_s:g} s time limit. Narrow it down.") from e
            raise ToolError(f"SQL error: {e}") from e
        except sqlite3.Error as e:
            raise ToolError(f"SQL error: {e}") from e
        truncated = len(rows) > MAX_SQL_ROWS
        result = {"columns": columns, "rows": [list(r) for r in rows[:MAX_SQL_ROWS]], "truncated": truncated}
        return json.dumps(result, default=str)


# ---------------- convert_currency ----------------

@dataclass
class CurrencyTool:
    api_url: str
    timeout: float = 10.0

    def __call__(self, amount: float, from_currency: str, to_currency: str) -> str:
        src, dst = from_currency.strip().upper(), to_currency.strip().upper()
        if not (re.fullmatch(r"[A-Z]{3}", src) and re.fullmatch(r"[A-Z]{3}", dst)):
            raise ToolError("Currencies must be 3-letter ISO codes, e.g. EUR, INR, USD.")
        if src == dst:
            return json.dumps({"amount": amount, "from": src, "to": dst, "rate": 1.0, "converted": amount})
        try:
            response = httpx.get(self.api_url, params={"base": src, "symbols": dst}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            rate = float(data["rates"][dst])
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise ToolError(f"Currency API failed: {e}") from e
        return json.dumps({"amount": amount, "from": src, "to": dst, "rate": rate,
                           "converted": round(amount * rate, 2), "rate_date": data.get("date")})


# ---------------- schemas shown to the LLM ----------------

def tool_schemas(sql_schema: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Semantic + keyword search over company policy documents. Returns relevant passages with ids like S1 to cite.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "A focused search query, e.g. 'annual leave carry over limit'."}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": f"Run ONE read-only SQLite SELECT on the HR/IT database. Max {MAX_SQL_ROWS} rows. Schema:\n{sql_schema}",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string", "description": "A single SELECT statement. Use LIKE for name matching."}},
                    "required": ["sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert an amount between currencies using today's exchange rate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "from_currency": {"type": "string", "description": "ISO code, e.g. EUR"},
                        "to_currency": {"type": "string", "description": "ISO code, e.g. INR"},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
        },
    ]
