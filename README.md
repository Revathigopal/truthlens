# TruthLens

A six-agent pipeline that checks the claims in a piece of text. Its rule is **incentives before content**: it works out what the source stands to gain before it looks at any evidence, and that judgement limits how confident the final verdict may be.

## How it works

| Step | Agent | Output contract | Sees the evidence? |
| --- | --- | --- | --- |
| 1 | Claim extractor | `ClaimSet` | no |
| 2 | Source profiler | `SourceProfile` | no |
| 3 | Incentive analyst | `IncentiveReport` (stakes, risk 0 to 1) | **no, by design** |
| 4 | Evidence assessor | `EvidenceReport` | yes |
| 5 | Challenger | `ChallengeReport` | yes |
| 6 | Synthesizer | `Verdict` | yes |

- Every hand-off is a Pydantic model (`src/truthlens/contracts.py`). A reply that fails validation is returned to the agent with the error, once; a second failure stops the run.
- After the synthesizer, `apply_guards` enforces three rules in code: confidence can never exceed `1 - 0.5 × incentive_risk`, every claim gets a verdict, and opinions are never labelled true or false.

## Quick start

```bash
pip install -e ".[dev]"
truthlens verify --file examples/supplement_ad.txt \
  --source "supplement brand press release" \
  --evidence examples/supplement_evidence.txt
```

```text
Source: supplement brand press release (interested_party)
Incentive risk: 0.85 -> confidence ceiling 0.57

[likely_false | 0.57] Our herbal capsule is guaranteed to cure joint pain in 7 days.
[likely_true | 0.57] A 2021 study of 40 adults found reduced stiffness after eight weeks.
[not_checkable | 0.00] It is the best supplement on the market.
[unverifiable | 0.30] Order today for a limited discount.

Overall: mixed (confidence 0.57)
```

Add `--json` for the full report, including every agent's output.

## Using a real model

The default backend is an offline rule-based stand-in (`src/truthlens/mock.py`) so the project runs with no setup. It uses keyword rules, not a model. For real verdicts choose a provider:

```bash
truthlens verify --text "..." --source "..." --provider ollama --model llama3.1
GROQ_API_KEY=... truthlens verify --text "..." --provider groq
OPENROUTER_API_KEY=... truthlens verify --text "..." --provider openrouter
```

All providers go through one small OpenAI-compatible client (`src/truthlens/llm.py`).

## Limits

- TruthLens does not search the web. It judges claims against the evidence snippets you supply; with none, claims come back `unverifiable`.
- The automated tests cover the pipeline, contracts and guards using the offline backend. The HTTP client is tested against a local server that speaks the chat-completions protocol, not against live providers. Output quality with a given model is not measured here.

## Tests

```bash
pytest -q
```
