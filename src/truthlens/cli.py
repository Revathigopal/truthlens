"""Command-line interface: ``truthlens verify --text "..." --source "..."``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agents import AgentError
from .llm import LLMError, get_llm
from .mock import mock_handler
from .pipeline import TruthLensPipeline, VerificationReport


def render(report: VerificationReport) -> str:
    claims = {claim.id: claim for claim in report.claims.claims}
    lines = [
        f"Source: {report.source_profile.name} ({report.source_profile.kind})",
        f"Incentive risk: {report.incentives.incentive_risk:.2f} -> confidence ceiling {report.confidence_ceiling:.2f}",
    ]
    for stake in report.incentives.stakes:
        lines.append(f"  stake [{stake.type}] {stake.description}")
    lines.append("")
    for item in report.verdict.claim_verdicts:
        lines.append(f"[{item.label} | {item.confidence:.2f}] {claims[item.claim_id].text}")
        lines.append(f"    {item.reason}")
    for challenge in report.challenges.challenges:
        lines.append(f"  objection to claim {challenge.claim_id}: {challenge.objection}")
    lines += [
        "",
        f"Overall: {report.verdict.overall_label} (confidence {report.verdict.overall_confidence:.2f})",
        report.verdict.summary,
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="truthlens", description="Verify claims, weighing incentives before content.")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="Run the six-agent pipeline on a piece of text.")
    group = verify.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="The text to verify.")
    group.add_argument("--file", type=Path, help="Read the text from a file.")
    verify.add_argument("--source", default="unknown", help="Who made the claims, e.g. 'supplement brand press release'.")
    verify.add_argument("--evidence", type=Path, help="Text file with one evidence snippet per line.")
    verify.add_argument("--provider", help="mock (default), ollama, groq, openrouter or openai.")
    verify.add_argument("--model", help="Model name for the chosen provider.")
    verify.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    evidence = []
    if args.evidence:
        evidence = [line.strip() for line in args.evidence.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        llm = get_llm(args.provider, args.model, mock_handler=mock_handler)
        report = TruthLensPipeline(llm).verify(text, source=args.source, evidence=evidence)
    except (LLMError, AgentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
