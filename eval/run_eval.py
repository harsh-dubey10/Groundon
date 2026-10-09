"""Evaluate retrieval (no LLM needed) and, optionally, end-to-end answers.

    python eval/run_eval.py              # retrieval metrics for 4 configurations
    python eval/run_eval.py --answers    # + answer correctness, abstention, citations (needs GROQ_API_KEY)

Document-level metrics (a chunk is relevant if it comes from a relevant_source):
  precision@k = relevant chunks in top-k / k
  recall@k    = relevant documents found in top-k / relevant documents
  MRR         = mean of 1 / rank of the first relevant chunk
Chunk-level metrics (stricter: the chunk must contain the expected answer text):
  ans_hit@k   = 1 if any top-k chunk contains every expected answer string
  ans_mrr     = mean of 1 / rank of the first answer-bearing chunk
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundon.config import get_settings  # noqa: E402
from groundon.generation.grounding import is_abstention  # noqa: E402
from groundon.generation.llm import LLMError  # noqa: E402
from groundon.services import build_services  # noqa: E402

CONFIGS = {
    "vector only": dict(hybrid=False, rerank=False),
    "hybrid (BM25+vector, RRF)": dict(hybrid=True, rerank=False),
    "vector + rerank": dict(hybrid=False, rerank=True),
    "hybrid + rerank": dict(hybrid=True, rerank=True),
}


def retrieval_metrics(services, items, k: int) -> dict[str, dict]:
    answerable = [i for i in items if i["answerable"]]
    out = {}
    for name, opts in CONFIGS.items():
        p = r = mrr = ans_hit = ans_mrr = 0.0
        for item in answerable:
            relevant = set(item["relevant_sources"])
            chunks = [c.chunk for c in services.retriever.retrieve(item["question"], top_k=k, **opts).chunks]
            sources = [c.source for c in chunks]
            p += sum(s in relevant for s in sources) / k
            r += len(relevant & set(sources)) / len(relevant)
            first = next((rank for rank, s in enumerate(sources, 1) if s in relevant), None)
            mrr += 1 / first if first else 0.0
            expected = [e.lower() for e in item["expected"]]
            first_ans = next((rank for rank, c in enumerate(chunks, 1)
                              if c.source in relevant and all(e in c.text.lower() for e in expected)), None)
            ans_hit += float(first_ans is not None)
            ans_mrr += 1 / first_ans if first_ans else 0.0
        n = len(answerable)
        out[name] = {"precision@k": p / n, "recall@k": r / n, "mrr": mrr / n,
                     "ans_hit@k": ans_hit / n, "ans_mrr": ans_mrr / n}
    return out


_UNICODE_FIXES = str.maketrans({"\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2007": " ",
                                "\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2019": "'"})


def normalize(text: str) -> str:
    """LLMs emit non-breaking spaces/hyphens ("31\u202fMarch"); compare on plain ASCII forms."""
    return text.translate(_UNICODE_FIXES).lower()


def answer_with_backoff(services, question, k, verify, attempts=4):
    for attempt in range(attempts):
        try:
            return services.rag.answer(question, top_k=k, verify=verify)
        except LLMError as e:
            if "429" not in str(e) or attempt == attempts - 1:
                raise
            time.sleep(20 * (attempt + 1))  # free-tier tokens-per-minute window


def answer_metrics(services, items, k: int, verify: bool, delay: float = 0.0) -> dict:
    rows = []
    for item in items:
        time.sleep(delay)
        t = time.perf_counter()
        try:
            ans = answer_with_backoff(services, item["question"], k, verify)
        except LLMError as e:
            print(f"  ! {item['question'][:70]:70} (LLM error, skipped: {str(e)[:60]})")
            continue
        text = normalize(ans.answer)
        abstained = ans.abstained or is_abstention(ans.answer)
        row = {"question": item["question"], "answerable": item["answerable"], "abstained": abstained,
               "llm_called": ans.llm_called, "latency_s": time.perf_counter() - t}
        if item["answerable"]:
            row["correct"] = not abstained and all(normalize(e) in text for e in item["expected"])
        else:
            row["correct"] = abstained
        if item.get("forbidden"):
            row["injection_resisted"] = not any(normalize(f) in text for f in item["forbidden"])
        if ans.citations and not abstained:
            row["citations_valid"] = ans.citations.ok
            cited_sources = {s.source for s in ans.sources if s.label in ans.citations.cited}
            row["cites_relevant_doc"] = bool(cited_sources & set(item["relevant_sources"]))
        if ans.verification:
            row["grounded"] = ans.verification.grounded
        rows.append(row)
        row["answer"] = ans.answer
        mark = "✓" if row["correct"] else "✗"
        print(f"  {mark} {item['question'][:70]:70} {'(abstained)' if abstained else ''}")

    def rate(key, subset=None):
        vals = [r[key] for r in (subset or rows) if r.get(key) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "answer accuracy (answerable)": rate("correct", answerable),
        "correct abstention (unanswerable)": rate("correct", unanswerable),
        "false abstention (answerable)": rate("abstained", answerable),
        "citations valid": rate("citations_valid"),
        "cites a relevant doc": rate("cites_relevant_doc"),
        "injection resisted": rate("injection_resisted"),
        "grounded (verifier)": rate("grounded"),
        "LLM calls skipped by abstain gate": (sum(not r["llm_called"] for r in rows) / len(rows), len(rows)),
        "mean latency (s)": (sum(r["latency_s"] for r in rows) / len(rows), len(rows)),
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--answers", action="store_true", help="also evaluate LLM answers (needs GROQ_API_KEY)")
    parser.add_argument("--verify", action="store_true", help="run the groundedness verifier on each answer")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions (rate limits)")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results.json")
    args = parser.parse_args()

    items = json.loads((ROOT / "eval" / "eval_set.json").read_text())
    services = build_services(get_settings())
    report: dict = {"k": args.k, "n_questions": len(items)}

    print(f"\nRetrieval @k={args.k} ({sum(i['answerable'] for i in items)} answerable questions)\n")
    print(f"{'configuration':28} {'P@k':>6} {'R@k':>6} {'MRR':>6} | {'AnsHit@k':>8} {'AnsMRR':>6}")
    report["retrieval"] = retrieval_metrics(services, items, args.k)
    for name, m in report["retrieval"].items():
        print(f"{name:28} {m['precision@k']:6.3f} {m['recall@k']:6.3f} {m['mrr']:6.3f} | {m['ans_hit@k']:8.3f} {m['ans_mrr']:6.3f}")

    if args.answers:
        if services.llm is None:
            sys.exit("\n--answers needs GROQ_API_KEY in .env")
        print(f"\nAnswers (hybrid + rerank, k={args.k}{', verified' if args.verify else ''})\n")
        summary, rows = answer_metrics(services, items, args.k, args.verify, args.delay)
        print()
        for name, (value, n) in summary.items():
            shown = "n/a" if value is None else (f"{value:.2f}" if "latency" in name else f"{value:.0%}")
            print(f"  {name:36} {shown:>6}  (n={n})")
        report["answers"] = {k: v[0] for k, v in summary.items()}
        report["answer_rows"] = rows

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
