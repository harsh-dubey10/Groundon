"""Post-generation checks: are the citations real, and is the answer supported?"""

import json
import re
from dataclasses import dataclass, field

from groundon.generation.llm import LLMClient, LLMError
from groundon.generation.prompts import NOT_FOUND, VERIFIER_SYSTEM, VERIFIER_USER

_CITATION = re.compile(r"\[(S\d+)(?:\s*,\s*(S\d+))*\]")
# Models trained on other citation styles emit 【S1】, 【S1†L3-L5】 or [S1†source]; map them to [S1].
_ALT_CITATION = re.compile(r"[【\[]\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*(?:†[^】\]]*)?[】\]]")


@dataclass
class CitationReport:
    cited: list[str]  # valid labels the answer cites, in order of first use
    invalid: list[str]  # labels cited that were never retrieved -> fabricated
    has_citations: bool
    is_abstention: bool

    @property
    def ok(self) -> bool:
        return not self.invalid and (self.has_citations or self.is_abstention)


@dataclass
class Verification:
    grounded: bool | None  # None when the check couldn't run
    unsupported_claims: list[str] = field(default_factory=list)
    error: str | None = None


def normalize_citations(answer: str) -> str:
    return _ALT_CITATION.sub(lambda m: "[" + re.sub(r"\s*[,;]\s*", ", ", m.group(1)) + "]", answer)


def extract_citations(answer: str) -> list[str]:
    labels: list[str] = []
    for match in _CITATION.finditer(answer):
        for label in re.findall(r"S\d+", match.group(0)):
            if label not in labels:
                labels.append(label)
    return labels


def is_abstention(answer: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


def check_citations(answer: str, valid_labels: set[str]) -> CitationReport:
    labels = extract_citations(answer)
    return CitationReport(
        cited=[l for l in labels if l in valid_labels],
        invalid=[l for l in labels if l not in valid_labels],
        has_citations=bool(labels),
        is_abstention=is_abstention(answer),
    )


def verify_groundedness(llm: LLMClient, model: str, documents: str, answer: str) -> Verification:
    """LLM-as-judge: a second, cheap model checks every claim against the sources."""
    if answer.strip().rstrip(".").lower() == NOT_FOUND.rstrip(".").lower():
        return Verification(grounded=True)  # a pure abstention makes no claims to check
    try:
        result = llm.chat(
            [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": VERIFIER_USER.format(documents=documents, answer=answer)},
            ],
            model=model,
            json_mode=True,
        )
        data = json.loads(result.content)
        return Verification(grounded=bool(data.get("grounded")), unsupported_claims=list(data.get("unsupported_claims") or []))
    except (LLMError, json.JSONDecodeError, AttributeError) as e:
        return Verification(grounded=None, error=str(e))
