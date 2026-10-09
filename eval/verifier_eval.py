"""Measure the groundedness verifier on known-good and known-bad answers.

    python eval/verifier_eval.py          # needs GROQ_API_KEY and eval/results.json from `run_eval.py --answers`
    python eval/verifier_eval.py --model openai/gpt-oss-120b   # judge with another model

Checking only real answers says how often the verifier raises false alarms, not whether it catches
hallucinations. So each correct answer from the last answer eval is judged three times:
  original      the real answer                                 -> should be grounded
  number swap   every number off by one (citations kept)          -> should be flagged
  added claim   a fabricated, cited policy sentence appended    -> should be flagged
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundon.config import get_settings  # noqa: E402
from groundon.generation.grounding import verify_groundedness  # noqa: E402
from groundon.generation.llm import LLMError  # noqa: E402
from groundon.generation.rag import format_documents, to_sources  # noqa: E402
from groundon.services import build_services  # noqa: E402

FABRICATED = [
    "Any exception to this rule must also be approved in writing by the CEO [S1].",
    "Employees who breach this rule lose two days of annual leave [S1].",
    "Compliance with this rule is reviewed by the board of directors in a quarterly audit [S1].",
]
_NUMBER = re.compile(r"(?<![S\d])\d+(?:[.,]\d+)?")  # skip the digits of [S1]-style labels


def swap_numbers(answer: str) -> str:
    def shift(m: re.Match) -> str:  # off-by-one, like a realistic misread: 8 -> 9, 3,500 -> 3,501
        text = m.group()
        if "." in text:
            return f"{float(text) + 1:.{len(text.split('.')[1])}f}"
        value = int(text.replace(",", "")) + 1
        return f"{value:,}" if "," in text else str(value)
    return _NUMBER.sub(shift, answer)


class DailyLimit(Exception):
    """The verifier model's tokens-per-day quota is spent; waiting minutes won't help."""


def judge(services, model: str, documents: str, answer: str, attempts: int = 6):
    for attempt in range(attempts):
        v = verify_groundedness(services.llm, model, documents, answer)
        if v.grounded is None and "tokens per day" in (v.error or ""):
            raise DailyLimit(v.error)
        if v.grounded is not None or "429" not in (v.error or "") or attempt == attempts - 1:
            return v
        time.sleep(20 * (attempt + 1))  # free-tier tokens-per-minute window


def summarize(cases: list[dict]) -> dict:
    summary = {}
    for kind in ("original", "number swap", "added claim"):
        judged = [c for c in cases if c["kind"] == kind and c["grounded"] is not None]
        errors = sum(c["kind"] == kind and c["grounded"] is None for c in cases)
        right = sum(c["grounded"] == c["expected_grounded"] for c in judged)
        summary[kind] = {"rate": right / len(judged) if judged else None, "n": len(judged), "errors": errors}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="verifier model (default: VERIFIER_MODEL from settings)")
    args = parser.parse_args()

    results = json.loads((ROOT / "eval" / "results.json").read_text())
    if "answer_rows" not in results:
        sys.exit("Run `python eval/run_eval.py --answers` first")
    k = results["k"]
    services = build_services(get_settings())
    if services.llm is None:
        sys.exit("needs GROQ_API_KEY in .env")
    model = args.model or services.settings.verifier_model

    # Resumable: free-tier daily token caps can stop a run part-way, so judged cases are kept and skipped.
    out = ROOT / "eval" / f"verifier_results_{model.split('/')[-1]}.json"
    done = json.loads(out.read_text())["cases"] if out.exists() else []
    cache = {(c["question"], c["kind"], c["answer"]): c for c in done if c["grounded"] is not None}

    def save(cases):
        out.write_text(json.dumps({"verifier_model": model, "summary": summarize(cases), "cases": cases}, indent=2))

    rows = [r for r in results["answer_rows"] if r["answerable"] and r["correct"]]
    cases, stopped = [], None
    for i, row in enumerate(rows):
        variants = {"original": (row["answer"], True),
                    "added claim": (f"{row['answer'].rstrip()} {FABRICATED[i % len(FABRICATED)]}", False)}
        swapped = swap_numbers(row["answer"])
        if swapped != row["answer"]:
            variants["number swap"] = (swapped, False)
        documents = None
        for kind, (answer, should_ground) in variants.items():
            if (row["question"], kind, answer) in cache:
                cases.append(cache[(row["question"], kind, answer)])
                continue
            if stopped:
                continue
            if documents is None:
                documents = format_documents(to_sources(services.retriever.retrieve(row["question"], top_k=k).chunks))
            time.sleep(2)
            try:
                v = judge(services, model, documents, answer)
            except DailyLimit as e:
                stopped = str(e)
                continue
            case = {"question": row["question"], "kind": kind, "answer": answer, "expected_grounded": should_ground,
                    "grounded": v.grounded, "unsupported_claims": v.unsupported_claims, "error": v.error}
            cases.append(case)
            save(cases + [c for c in done if c not in cases])
            mark = "?" if v.grounded is None else ("✓" if v.grounded == should_ground else "✗")
            print(f"  {mark} {kind:12} {row['question'][:65]}")

    save(cases)
    print()
    for kind, m in summarize(cases).items():
        label = "passed as grounded" if kind == "original" else "flagged as ungrounded"
        rate = "n/a" if m["rate"] is None else f"{m['rate']:.0%}"
        print(f"  {kind:12} {label:22} {rate:>5}  (n={m['n']}, verifier errors={m['errors']})")
    print(f"\nSaved {out}")
    if stopped:
        sys.exit(f"\nStopped early, daily token limit for {model}. Re-run later to continue.\n{stopped[:200]}")


if __name__ == "__main__":
    main()
