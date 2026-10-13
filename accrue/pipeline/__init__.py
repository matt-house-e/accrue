"""Pipeline execution engine for the Accrue enrichment engine."""

from .compare import ComparisonResult, compare
from .merge import Candidate, MergedResult, merge_runs
from .pipeline import Pipeline, PipelineResult
from .plan import PipelinePlan, StepPlan

__all__ = [
    "Pipeline",
    "PipelineResult",
    "PipelinePlan",
    "StepPlan",
    "compare",
    "ComparisonResult",
    "merge_runs",
    "MergedResult",
    "Candidate",
]
