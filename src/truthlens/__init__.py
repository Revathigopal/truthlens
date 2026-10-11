"""TruthLens: a six-agent claim verification pipeline that weighs incentives before content."""

from .contracts import Verdict
from .pipeline import TruthLensPipeline, VerificationReport

__all__ = ["TruthLensPipeline", "VerificationReport", "Verdict"]
__version__ = "0.1.0"
