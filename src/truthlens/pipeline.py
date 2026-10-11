"""Pipeline orchestration.

The ordering is the point of the project: the source's incentives are
analysed *before* any agent looks at evidence, and the resulting risk score
then limits how confident the final verdict is allowed to be.
"""

from __future__ import annotations

from pydantic import BaseModel

from .agents import (
    CHALLENGER,
    CLAIM_EXTRACTOR,
    EVIDENCE_ASSESSOR,
    INCENTIVE_ANALYST,
    SOURCE_PROFILER,
    SYNTHESIZER,
)
from .contracts import (
    ChallengeReport,
    ClaimSet,
    ClaimVerdict,
    EvidenceReport,
    IncentiveReport,
    SourceProfile,
    Verdict,
)
from .llm import LLM

# With maximum incentive risk, confidence can be at most 1 - INCENTIVE_DISCOUNT.
INCENTIVE_DISCOUNT = 0.5


class VerificationReport(BaseModel):
    text: str
    source: str
    claims: ClaimSet
    source_profile: SourceProfile
    incentives: IncentiveReport
    evidence: EvidenceReport
    challenges: ChallengeReport
    verdict: Verdict
    confidence_ceiling: float
    trace: list[str]


def confidence_ceiling(incentive_risk: float) -> float:
    """Highest confidence allowed for a source with the given incentive risk."""
    return round(1.0 - INCENTIVE_DISCOUNT * incentive_risk, 4)


def apply_guards(verdict: Verdict, claims: ClaimSet, incentives: IncentiveReport) -> Verdict:
    """Enforce in code what the synthesizer was asked to do in its prompt.

    1. No confidence may exceed the ceiling set by incentive risk.
    2. Every claim gets exactly one verdict; missing ones become 'unverifiable'.
    3. Claims that are not checkable can never be labelled true or false.
    """
    ceiling = confidence_ceiling(incentives.incentive_risk)
    by_id = {item.claim_id: item for item in verdict.claim_verdicts}
    guarded: list[ClaimVerdict] = []
    for claim in claims.claims:
        item = by_id.get(claim.id)
        if item is None:
            item = ClaimVerdict(
                claim_id=claim.id,
                label="unverifiable",
                confidence=0.0,
                reason="No verdict was produced for this claim.",
            )
        if not claim.checkable and item.label != "not_checkable":
            item = item.model_copy(
                update={"label": "not_checkable", "reason": "Opinion or value judgement; not a factual claim."}
            )
        if item.confidence > ceiling:
            item = item.model_copy(update={"confidence": ceiling})
        guarded.append(item)
    return verdict.model_copy(
        update={
            "claim_verdicts": guarded,
            "overall_confidence": min(verdict.overall_confidence, ceiling),
        }
    )


class TruthLensPipeline:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def verify(self, text: str, source: str = "unknown", evidence: list[str] | None = None) -> VerificationReport:
        evidence = evidence or []
        trace: list[str] = []

        claims = CLAIM_EXTRACTOR.run(self.llm, {"text": text})
        trace.append(CLAIM_EXTRACTOR.name)

        profile = SOURCE_PROFILER.run(self.llm, {"source": source, "text": text})
        trace.append(SOURCE_PROFILER.name)

        # Incentives first: this agent never sees the evidence.
        incentives = INCENTIVE_ANALYST.run(
            self.llm, {"source_profile": profile.model_dump(), "claims": claims.model_dump()}
        )
        trace.append(INCENTIVE_ANALYST.name)

        evidence_report = EVIDENCE_ASSESSOR.run(
            self.llm,
            {
                "claims": claims.model_dump(),
                "evidence_snippets": evidence,
                "incentive_risk": incentives.incentive_risk,
            },
        )
        trace.append(EVIDENCE_ASSESSOR.name)

        challenges = CHALLENGER.run(
            self.llm, {"claims": claims.model_dump(), "evidence": evidence_report.model_dump()}
        )
        trace.append(CHALLENGER.name)

        raw_verdict = SYNTHESIZER.run(
            self.llm,
            {
                "claims": claims.model_dump(),
                "incentives": incentives.model_dump(),
                "evidence": evidence_report.model_dump(),
                "challenges": challenges.model_dump(),
            },
        )
        trace.append(SYNTHESIZER.name)

        return VerificationReport(
            text=text,
            source=source,
            claims=claims,
            source_profile=profile,
            incentives=incentives,
            evidence=evidence_report,
            challenges=challenges,
            verdict=apply_guards(raw_verdict, claims, incentives),
            confidence_ceiling=confidence_ceiling(incentives.incentive_risk),
            trace=trace,
        )
