"""Accrue — Enrichment Pipeline Engine.

A programmatic enrichment engine built on composable, column-oriented
pipeline steps with Pydantic validation, checkpointing, and async concurrency.
"""

import logging as _logging

_logging.getLogger("accrue").addHandler(_logging.NullHandler())

# Public API
from .core import (
    Enricher,
    EnrichmentConfig,
    EnrichmentError,
    FieldValidationError,
    PipelineError,
    RowError,
    StepError,
)
from .core.hooks import (
    EnrichmentHooks,
    PipelineEndEvent,
    PipelineStartEvent,
    RowCompleteEvent,
    StepEndEvent,
    StepStartEvent,
)
from .pipeline import Pipeline, PipelinePlan, PipelineResult, StepPlan
from .schemas.base import CostSummary
from .schemas.field_spec import FieldSpec
from .schemas.grounding import GroundingConfig
from .steps import FunctionStep, LLMStep, Step, StepContext, StepResult
from .steps.providers.base import BatchCapableLLMClient, BatchRequest, BatchResult
from .utils.logger import setup_logging
from .utils.web_search import web_search

__version__ = "1.3.0"
__author__ = "Accrue Team"

__all__ = [
    # Primary API
    "Pipeline",
    "PipelineResult",
    "PipelinePlan",
    "StepPlan",
    "LLMStep",
    "FunctionStep",
    "EnrichmentConfig",
    # Hooks
    "EnrichmentHooks",
    "PipelineStartEvent",
    "PipelineEndEvent",
    "StepStartEvent",
    "StepEndEvent",
    "RowCompleteEvent",
    # Step protocol
    "Step",
    "StepContext",
    "StepResult",
    # Schemas
    "FieldSpec",
    "GroundingConfig",
    # Utilities
    "setup_logging",
    "web_search",
    # Batch API
    "BatchCapableLLMClient",
    "BatchRequest",
    "BatchResult",
    # Results & errors
    "CostSummary",
    "RowError",
    "EnrichmentError",
    "FieldValidationError",
    "StepError",
    "PipelineError",
    # Internal runner (power users)
    "Enricher",
]
