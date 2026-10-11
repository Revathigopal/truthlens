"""Typed contracts passed between agents.

Every agent returns one of these models. A reply that does not validate is
rejected and the agent is asked again, so malformed output never travels
down the pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Score = float


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(Contract):
    id: int = Field(ge=1)
    text: str = Field(min_length=3)
    checkable: bool = Field(description="False for opinions, slogans and value judgements.")


class ClaimSet(Contract):
    claims: list[Claim] = Field(min_length=1)


class SourceProfile(Contract):
    name: str
    kind: Literal["interested_party", "academic", "official", "news", "anonymous", "unknown"]
    relationship_to_subject: str


class Stake(Contract):
    type: Literal["financial", "reputational", "ideological", "political"]
    description: str


class IncentiveReport(Contract):
    stakes: list[Stake]
    incentive_risk: Score = Field(ge=0, le=1, description="0 = no reason to mislead, 1 = strong reason.")
    rationale: str


class EvidenceItem(Contract):
    claim_id: int
    support: Literal["supported", "contradicted", "unverifiable"]
    specificity: Score = Field(ge=0, le=1)
    notes: str


class EvidenceReport(Contract):
    items: list[EvidenceItem]


class Challenge(Contract):
    claim_id: int
    objection: str
    severity: Score = Field(ge=0, le=1)


class ChallengeReport(Contract):
    challenges: list[Challenge]


Label = Literal["likely_true", "uncertain", "likely_false", "unverifiable", "not_checkable"]


class ClaimVerdict(Contract):
    claim_id: int
    label: Label
    confidence: Score = Field(ge=0, le=1)
    reason: str


class Verdict(Contract):
    claim_verdicts: list[ClaimVerdict]
    overall_label: Literal["credible", "mixed", "doubtful", "insufficient_evidence"]
    overall_confidence: Score = Field(ge=0, le=1)
    summary: str
