"""The six agents. Each one has a single job and a single output contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import (
    ChallengeReport,
    ClaimSet,
    EvidenceReport,
    IncentiveReport,
    SourceProfile,
    Verdict,
)
from .llm import LLM, extract_json

T = TypeVar("T", bound=BaseModel)


class AgentError(RuntimeError):
    """An agent failed to produce output that satisfies its contract."""


@dataclass(frozen=True)
class Agent(Generic[T]):
    name: str
    instructions: str
    output_model: type[T]
    max_attempts: int = 2

    def system_prompt(self) -> str:
        schema = json.dumps(self.output_model.model_json_schema(), indent=2)
        return (
            f"AGENT:{self.name}\n{self.instructions}\n\n"
            "Reply with exactly one JSON object and nothing else. "
            f"It must satisfy this JSON schema:\n{schema}"
        )

    def run(self, llm: LLM, payload: dict) -> T:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error = ""
        for _ in range(self.max_attempts):
            reply = llm.complete(messages)
            try:
                return self.output_model.model_validate(extract_json(reply))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": f"That reply was rejected: {last_error}\nSend a corrected JSON object.",
                    },
                ]
        raise AgentError(f"{self.name} failed its contract after {self.max_attempts} attempts: {last_error}")


CLAIM_EXTRACTOR = Agent(
    name="claim_extractor",
    instructions=(
        "Split the text into separate claims. Keep each claim as close to the original "
        "wording as possible. Mark a claim as checkable only if evidence could in principle "
        "confirm or refute it; opinions, slogans and value judgements are not checkable."
    ),
    output_model=ClaimSet,
)

SOURCE_PROFILER = Agent(
    name="source_profiler",
    instructions=(
        "Describe who is making these claims and how they relate to the subject. "
        "Use 'interested_party' when the source sells, owns, represents or campaigns for "
        "the thing the claims are about. Do not judge whether the claims are true."
    ),
    output_model=SourceProfile,
)

INCENTIVE_ANALYST = Agent(
    name="incentive_analyst",
    instructions=(
        "List what the source stands to gain or lose if readers believe these claims. "
        "Score incentive_risk from 0 (no reason to mislead) to 1 (strong reason to mislead). "
        "Judge incentives only; you have not been shown any evidence and must not assess truth."
    ),
    output_model=IncentiveReport,
)

EVIDENCE_ASSESSOR = Agent(
    name="evidence_assessor",
    instructions=(
        "For each checkable claim, compare it with the supplied evidence snippets. "
        "Use 'supported' or 'contradicted' only when a snippet directly addresses the claim; "
        "otherwise use 'unverifiable'. Score specificity by how precise and attributable the "
        "claim is. The higher the incentive_risk you are given, the more direct the evidence "
        "must be before you call a claim supported."
    ),
    output_model=EvidenceReport,
)

CHALLENGER = Agent(
    name="challenger",
    instructions=(
        "Argue against the claims. For each claim that deserves it, give the strongest "
        "objection: an overstated absolute, a missing comparison, an unstated source, or a "
        "conflict with the evidence. Score severity from 0 to 1. Skip claims with no real objection."
    ),
    output_model=ChallengeReport,
)

SYNTHESIZER = Agent(
    name="synthesizer",
    instructions=(
        "Give a verdict for every claim using the incentive report, evidence report and "
        "challenges. Claims marked not checkable get the label 'not_checkable'. Confidence "
        "must fall as incentive_risk rises. Then give an overall label and a two-sentence summary."
    ),
    output_model=Verdict,
)

ALL_AGENTS = (
    CLAIM_EXTRACTOR,
    SOURCE_PROFILER,
    INCENTIVE_ANALYST,
    EVIDENCE_ASSESSOR,
    CHALLENGER,
    SYNTHESIZER,
)
