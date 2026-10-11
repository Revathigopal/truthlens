import json

import pytest

from truthlens.agents import CLAIM_EXTRACTOR, AgentError
from truthlens.cli import main
from truthlens.contracts import ClaimSet, ClaimVerdict, IncentiveReport, Verdict
from truthlens.llm import MockLLM, extract_json
from truthlens.mock import mock_handler
from truthlens.pipeline import TruthLensPipeline, apply_guards, confidence_ceiling

AD = (
    "Our herbal capsule is guaranteed to cure joint pain in 7 days. "
    "A 2021 study of 40 adults found reduced stiffness after eight weeks. "
    "It is the best supplement on the market."
)
EVIDENCE = [
    "A 2021 study of 40 adults found reduced stiffness after eight weeks of use.",
    "Regulators found no evidence that any herbal capsule can cure joint pain in 7 days.",
]


def run(source="supplement brand press release", evidence=EVIDENCE):
    return TruthLensPipeline(MockLLM(mock_handler)).verify(AD, source=source, evidence=evidence)


def test_six_agents_run_in_order_with_incentives_before_evidence():
    report = run()
    assert report.trace == [
        "claim_extractor",
        "source_profiler",
        "incentive_analyst",
        "evidence_assessor",
        "challenger",
        "synthesizer",
    ]
    assert report.trace.index("incentive_analyst") < report.trace.index("evidence_assessor")


def test_incentive_analyst_is_never_shown_evidence():
    llm = MockLLM(mock_handler)
    TruthLensPipeline(llm).verify(AD, source="brand", evidence=EVIDENCE)
    incentive_call = next(c for c in llm.calls if c[0]["content"].startswith("AGENT:incentive_analyst"))
    assert "evidence_snippets" not in incentive_call[1]["content"]
    assert "Regulators" not in incentive_call[1]["content"]


def test_verdicts_follow_the_evidence():
    report = run()
    labels = {v.claim_id: v.label for v in report.verdict.claim_verdicts}
    assert labels == {1: "likely_false", 2: "likely_true", 3: "not_checkable"}
    assert report.verdict.overall_label == "mixed"


def test_interested_source_lowers_confidence_ceiling():
    brand = run(source="supplement brand press release")
    journal = run(source="peer-reviewed journal")
    assert brand.incentives.incentive_risk > journal.incentives.incentive_risk
    assert brand.confidence_ceiling < journal.confidence_ceiling
    assert all(v.confidence <= brand.confidence_ceiling for v in brand.verdict.claim_verdicts)


def test_guard_caps_overconfident_synthesizer():
    claims = ClaimSet.model_validate(
        {"claims": [{"id": 1, "text": "Sales doubled.", "checkable": True}, {"id": 2, "text": "It is great.", "checkable": False}]}
    )
    incentives = IncentiveReport(stakes=[], incentive_risk=1.0, rationale="max")
    raw = Verdict(
        claim_verdicts=[ClaimVerdict(claim_id=2, label="likely_true", confidence=0.99, reason="x")],
        overall_label="credible",
        overall_confidence=0.99,
        summary="s",
    )
    guarded = apply_guards(raw, claims, incentives)
    assert confidence_ceiling(1.0) == 0.5
    assert guarded.overall_confidence == 0.5
    by_id = {v.claim_id: v for v in guarded.claim_verdicts}
    assert by_id[1].label == "unverifiable"  # missing verdict filled in
    assert by_id[2].label == "not_checkable"  # opinion cannot be 'true'
    assert by_id[2].confidence == 0.5


def test_agent_retries_then_succeeds_on_bad_json():
    replies = iter(["sorry, here you go", '```json\n{"claims": [{"id": 1, "text": "Water boils.", "checkable": true}]}\n```'])
    llm = MockLLM(lambda messages: next(replies))
    result = CLAIM_EXTRACTOR.run(llm, {"text": "Water boils."})
    assert result.claims[0].text == "Water boils."
    assert len(llm.calls) == 2
    assert "rejected" in llm.calls[1][-1]["content"]


def test_agent_fails_after_repeated_contract_violations():
    llm = MockLLM(lambda messages: '{"claims": []}')
    with pytest.raises(AgentError):
        CLAIM_EXTRACTOR.run(llm, {"text": "anything"})


def test_extract_json_handles_braces_inside_strings():
    assert extract_json('noise {"a": "x } y", "b": {"c": 1}} tail') == {"a": "x } y", "b": {"c": 1}}


def test_cli_json_output(capsys):
    code = main(["verify", "--text", AD, "--source", "brand advert", "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"]["overall_label"] == "insufficient_evidence"
    assert report["source_profile"]["kind"] == "interested_party"
