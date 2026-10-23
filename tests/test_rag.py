import json

from groundon.generation.grounding import check_citations, extract_citations, normalize_citations
from groundon.generation.llm import ChatResult
from groundon.generation.prompts import NOT_FOUND
from tests.conftest import ScriptedLLM


def test_extract_citations_handles_groups():
    assert extract_citations("A [S1]. B [S2, S3]. C [S1].") == ["S1", "S2", "S3"]


def test_alternative_citation_styles_are_normalized():
    text = "8 days【S1】 by March【S2†L3-L5】 and more [S3†leave.pdf] and 【S1, S4】."
    assert normalize_citations(text) == "8 days[S1] by March[S2] and more [S3] and [S1, S4]."
    assert normalize_citations("plain [S1] stays") == "plain [S1] stays"


def test_fabricated_citation_is_flagged():
    report = check_citations("You get 24 days [S1] and a bonus [S9].", {"S1", "S2"})
    assert report.invalid == ["S9"] and not report.ok


def test_uncited_answer_is_not_ok():
    assert not check_citations("You get 24 days.", {"S1"}).ok
    assert check_citations(NOT_FOUND, {"S1"}).ok  # abstaining needs no citation


def test_grounded_answer_flow(make_services):
    llm = ScriptedLLM([ChatResult(content="You can carry over up to 8 days [S1].")])
    s = make_services(llm)
    ans = s.rag.answer("How many leave days can be carried over?", top_k=2, rerank=False)

    assert ans.llm_called and not ans.abstained
    assert ans.citations.ok and ans.citations.cited == ["S1"]
    assert ans.sources[0].source == "leave.md"
    prompt = llm.calls[0]["messages"][1]["content"]
    assert '<document id="S1"' in prompt and "carried over" in prompt


def test_low_confidence_abstains_without_calling_llm(make_services):
    llm = ScriptedLLM()
    s = make_services(llm)
    ans = s.rag.answer("quantum chromodynamics lattice", rerank=False)
    assert ans.abstained and not ans.llm_called
    assert ans.answer == NOT_FOUND
    assert llm.calls == []  # saved an API call


def test_verifier_runs_on_cheap_model(make_services):
    llm = ScriptedLLM([
        ChatResult(content="Per diem in Germany is EUR 75 [S1]."),
        ChatResult(content=json.dumps({"grounded": False, "unsupported_claims": ["EUR 75"]})),
    ])
    s = make_services(llm)
    ans = s.rag.answer("per diem Germany", rerank=False, verify=True)
    assert ans.verification.grounded is False
    assert ans.verification.unsupported_claims == ["EUR 75"]
    assert llm.calls[1]["model"] == s.settings.verifier_model and llm.calls[1]["json_mode"]


def test_verifier_failure_is_reported_not_raised(make_services):
    llm = ScriptedLLM([ChatResult(content="EUR 75 [S1]."), ChatResult(content="not json")])
    ans = make_services(llm).rag.answer("per diem Germany", rerank=False, verify=True)
    assert ans.verification.grounded is None and ans.verification.error


def test_pure_abstention_skips_verifier(make_services):
    llm = ScriptedLLM([ChatResult(content=NOT_FOUND)])
    ans = make_services(llm).rag.answer("per diem Germany", rerank=False, verify=True)
    assert ans.abstained and ans.verification.grounded is True
    assert len(llm.calls) == 1  # no verifier call


def test_verifier_string_verdict_is_not_trusted(make_services):
    # "false" as a string is truthy in Python; a malformed verdict must not count as grounded.
    llm = ScriptedLLM([
        ChatResult(content="Per diem in Germany is EUR 75 [S1]."),
        ChatResult(content=json.dumps({"grounded": "false", "unsupported_claims": []})),
    ])
    ans = make_services(llm).rag.answer("per diem Germany", rerank=False, verify=True)
    assert ans.verification.grounded is None and ans.verification.error


def test_verifier_single_claim_string_is_kept_whole(make_services):
    llm = ScriptedLLM([
        ChatResult(content="Per diem in Germany is EUR 90 [S1]."),
        ChatResult(content=json.dumps({"grounded": False, "unsupported_claims": "EUR 90"})),
    ])
    ans = make_services(llm).rag.answer("per diem Germany", rerank=False, verify=True)
    assert ans.verification.grounded is False
    assert ans.verification.unsupported_claims == ["EUR 90"]
