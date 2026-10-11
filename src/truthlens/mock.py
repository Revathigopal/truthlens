"""Offline rule-based stand-in for a language model.

This is NOT a model. It answers each agent with simple keyword heuristics so
that the pipeline, its contracts and its guards can be run and tested with no
network and no API key. Use a real provider for meaningful verdicts.
"""

from __future__ import annotations

import json
import re

from .llm import Message

OPINION_WORDS = {"best", "amazing", "love", "beautiful", "awesome", "worst", "believe", "feel", "greatest"}
ABSOLUTE_WORDS = {"always", "never", "guaranteed", "guarantee", "100%", "cure", "cures", "everyone", "nobody", "proven"}
SALES_WORDS = {"buy", "offer", "discount", "invest", "returns", "order", "subscribe", "limited"}
CITATION_MARKERS = ("according to", "study", "journal", "doi", "http", "survey", "report")
NEGATIONS = ("no evidence", "not ", "false", "did not", "does not", "failed", "debunked")
STOPWORDS = {"the", "a", "an", "of", "and", "is", "are", "in", "to", "that", "this", "it", "for", "on", "with", "by", "our", "was"}

SOURCE_KINDS = [
    ("interested_party", ("company", "brand", "seller", "press release", "advert", "founder", "manufacturer", "campaign")),
    ("academic", ("journal", "university", "study", "researcher", "professor")),
    ("official", ("government", "ministry", "regulator", "court", "census")),
    ("news", ("newspaper", "news", "reporter", "magazine")),
    ("anonymous", ("anonymous", "forward", "whatsapp", "unknown blog", "viral")),
]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9%]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def _claim_extractor(payload: dict) -> dict:
    claims = []
    for index, sentence in enumerate(_sentences(payload["text"]) or [payload["text"]], start=1):
        words = _words(sentence)
        opinion = bool(words & OPINION_WORDS) and not re.search(r"\d", sentence)
        claims.append({"id": index, "text": sentence, "checkable": not opinion})
    return {"claims": claims}


def _source_profiler(payload: dict) -> dict:
    source = payload["source"]
    lowered = source.lower()
    kind = "unknown"
    for candidate, markers in SOURCE_KINDS:
        if any(marker in lowered for marker in markers):
            kind = candidate
            break
    relation = {
        "interested_party": "Benefits directly if the claims are believed.",
        "academic": "Research source with no stated commercial link to the subject.",
        "official": "Public body reporting on the subject.",
        "news": "Reports on the subject as a third party.",
        "anonymous": "Origin cannot be traced.",
        "unknown": "Relationship to the subject is not stated.",
    }[kind]
    return {"name": source, "kind": kind, "relationship_to_subject": relation}


def _incentive_analyst(payload: dict) -> dict:
    kind = payload["source_profile"]["kind"]
    text = " ".join(claim["text"] for claim in payload["claims"]["claims"])
    words = _words(text)
    base = {"interested_party": 0.6, "anonymous": 0.5, "unknown": 0.4, "news": 0.2, "official": 0.15, "academic": 0.1}[kind]
    stakes = []
    if kind == "interested_party":
        stakes.append({"type": "financial", "description": "The source gains commercially if the claims are believed."})
    if words & SALES_WORDS:
        base += 0.25
        stakes.append({"type": "financial", "description": "The text asks the reader to buy, invest or sign up."})
    if kind == "anonymous":
        stakes.append({"type": "reputational", "description": "An untraceable source bears no cost for being wrong."})
    risk = round(min(base, 1.0), 2)
    return {
        "stakes": stakes,
        "incentive_risk": risk,
        "rationale": f"Source type '{kind}' with {len(stakes)} identified stake(s).",
    }


def _evidence_assessor(payload: dict) -> dict:
    snippets = payload.get("evidence_snippets", [])
    needed_overlap = 3 if payload.get("incentive_risk", 0) >= 0.5 else 2
    items = []
    for claim in payload["claims"]["claims"]:
        if not claim["checkable"]:
            continue
        text = claim["text"]
        lowered = text.lower()
        specificity = 0.3
        if re.search(r"\d", text):
            specificity += 0.3
        if any(marker in lowered for marker in CITATION_MARKERS):
            specificity += 0.3
        support, notes = "unverifiable", "No supplied evidence addresses this claim."
        claim_words = _words(text)
        for snippet in snippets:
            if len(claim_words & _words(snippet)) >= needed_overlap:
                if any(neg in snippet.lower() for neg in NEGATIONS):
                    support, notes = "contradicted", f"Evidence disputes this: {snippet[:120]}"
                else:
                    support, notes = "supported", f"Evidence agrees: {snippet[:120]}"
                break
        items.append(
            {"claim_id": claim["id"], "support": support, "specificity": round(min(specificity, 1.0), 2), "notes": notes}
        )
    return {"items": items}


def _challenger(payload: dict) -> dict:
    support = {item["claim_id"]: item["support"] for item in payload["evidence"]["items"]}
    challenges = []
    for claim in payload["claims"]["claims"]:
        if not claim["checkable"]:
            continue
        lowered = claim["text"].lower()
        hits = sorted(word for word in ABSOLUTE_WORDS if word in lowered)
        if hits:
            challenges.append(
                {
                    "claim_id": claim["id"],
                    "objection": f"Uses the absolute term '{hits[0]}', which a single counter-example would refute.",
                    "severity": 0.7,
                }
            )
        elif support.get(claim["id"]) == "unverifiable" and not any(m in lowered for m in CITATION_MARKERS):
            challenges.append(
                {"claim_id": claim["id"], "objection": "States a fact without naming where it comes from.", "severity": 0.4}
            )
    return {"challenges": challenges}


def _synthesizer(payload: dict) -> dict:
    risk = payload["incentives"]["incentive_risk"]
    evidence = {item["claim_id"]: item for item in payload["evidence"]["items"]}
    severity = {}
    for challenge in payload["challenges"]["challenges"]:
        severity[challenge["claim_id"]] = max(severity.get(challenge["claim_id"], 0), challenge["severity"])
    verdicts = []
    for claim in payload["claims"]["claims"]:
        cid = claim["id"]
        if not claim["checkable"]:
            verdicts.append({"claim_id": cid, "label": "not_checkable", "confidence": 0.0, "reason": "Opinion, not a factual claim."})
            continue
        item = evidence.get(cid, {"support": "unverifiable", "specificity": 0.3})
        doubt = severity.get(cid, 0.0)
        if item["support"] == "supported":
            label, confidence = "likely_true", 0.85 - 0.3 * doubt
            reason = "Supplied evidence supports the claim."
        elif item["support"] == "contradicted":
            label, confidence = "likely_false", 0.8
            reason = "Supplied evidence contradicts the claim."
        elif doubt >= 0.6 and risk >= 0.5:
            label, confidence = "uncertain", 0.4
            reason = "Overstated claim from a source with a reason to mislead, and no evidence either way."
        else:
            label, confidence = "unverifiable", 0.3
            reason = "Nothing supplied confirms or refutes the claim."
        verdicts.append({"claim_id": cid, "label": label, "confidence": round(max(confidence, 0.0), 2), "reason": reason})

    labels = [v["label"] for v in verdicts if v["label"] != "not_checkable"]
    true_count, false_count = labels.count("likely_true"), labels.count("likely_false")
    if not labels or (true_count == 0 and false_count == 0):
        overall = "insufficient_evidence"
    elif false_count == 0 and true_count == len(labels):
        overall = "credible"
    elif false_count > true_count:
        overall = "doubtful"
    else:
        overall = "mixed"
    scored = [v["confidence"] for v in verdicts if v["label"] != "not_checkable"]
    overall_confidence = round(sum(scored) / len(scored), 2) if scored else 0.0
    summary = (
        f"{true_count} of {len(labels)} checkable claim(s) are supported and {false_count} contradicted. "
        f"Source incentive risk is {risk:.2f}."
    )
    return {
        "claim_verdicts": verdicts,
        "overall_label": overall,
        "overall_confidence": overall_confidence,
        "summary": summary,
    }


HANDLERS = {
    "claim_extractor": _claim_extractor,
    "source_profiler": _source_profiler,
    "incentive_analyst": _incentive_analyst,
    "evidence_assessor": _evidence_assessor,
    "challenger": _challenger,
    "synthesizer": _synthesizer,
}


def mock_handler(messages: list[Message]) -> str:
    """Route a request to the heuristic for the agent named in the system prompt."""
    first_line = messages[0]["content"].splitlines()[0]
    agent_name = first_line.removeprefix("AGENT:").strip()
    payload = json.loads(messages[1]["content"])
    return json.dumps(HANDLERS[agent_name](payload))
