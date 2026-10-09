import json

import pytest

from groundon.agent.tools import SqlTool, ToolError
from groundon.generation.llm import ChatResult
from tests.conftest import ScriptedLLM, tool_call


@pytest.fixture
def sql(make_services, settings):
    make_services()  # seeds the company DB
    return SqlTool(settings.company_db_path)


def test_sql_select_works(sql):
    result = json.loads(sql("SELECT name FROM employees WHERE name LIKE '%Priya%'"))
    assert result["rows"] == [["Priya Sharma"]]


@pytest.mark.parametrize("statement", [
    "DELETE FROM employees",
    "DROP TABLE employees",
    "UPDATE employees SET name = 'x'",
    "INSERT INTO employees VALUES (99,'a','b','c','d','e',NULL,'2020-01-01')",
    "SELECT 1; DROP TABLE employees",
    "ATTACH DATABASE '/tmp/x.db' AS x",
    "PRAGMA writable_schema = 1",
])
def test_sql_writes_are_blocked(sql, statement):
    with pytest.raises(ToolError):
        sql(statement)
    assert json.loads(sql("SELECT COUNT(*) FROM employees"))["rows"] == [[10]]


def test_sql_readonly_even_if_regex_bypassed(sql):
    # A CTE starts with WITH, passing the regex, but the write is still refused by SQLite.
    with pytest.raises(ToolError):
        sql("WITH x AS (SELECT 1) DELETE FROM employees")


def test_agent_combines_sql_and_documents(make_services):
    llm = ScriptedLLM([
        tool_call("query_database", sql="SELECT annual_total - annual_used FROM leave_balances WHERE employee_id = 2"),
        tool_call("search_documents", query="leave carry over"),
        ChatResult(content="Priya has 19 days left; up to 8 can be carried over [S1]."),
    ])
    s = make_services(llm)
    ans = s.agent.run("How many days does Priya have left and how many carry over?")

    assert [st.tool for st in ans.steps] == ["query_database", "search_documents"]
    assert json.loads(ans.steps[0].result)["rows"] == [[19]]
    assert ans.stopped_reason == "answered" and ans.citations.ok
    # Tool results were fed back to the model as role=tool messages.
    final_request = llm.calls[-1]["messages"]
    assert [m["role"] for m in final_request].count("tool") == 2


def test_agent_sees_tool_errors_and_can_recover(make_services):
    llm = ScriptedLLM([
        tool_call("query_database", sql="DROP TABLE employees"),
        tool_call("query_database", sql="SELECT COUNT(*) FROM employees"),
        ChatResult(content="There are 10 employees per the HR database."),
    ])
    ans = make_services(llm).agent.run("How many employees?")
    assert ans.steps[0].error and "read-only" in ans.steps[0].result
    assert not ans.steps[1].error


def test_agent_unknown_tool_and_step_limit(make_services):
    llm = ScriptedLLM([tool_call("launch_rockets")] * 6 + [ChatResult(content="I couldn't complete this.")])
    s = make_services(llm)
    ans = s.agent.run("do something")
    assert ans.stopped_reason == "max_steps"
    assert all(st.error for st in ans.steps)
    assert llm.calls[-1]["tools"] is None  # final call forces an answer without tools


def test_currency_tool_validates_codes(make_services):
    from groundon.agent.tools import CurrencyTool

    tool = CurrencyTool("http://currency.invalid")
    with pytest.raises(ToolError):
        tool(10, "euro", "INR")
    assert json.loads(tool(5, "INR", "inr"))["converted"] == 5
