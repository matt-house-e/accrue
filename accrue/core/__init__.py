"""Core functionality for the Accrue enrichment tool."""

from .cache import CacheManager
from .checkpoint import CheckpointData, CheckpointManager
from .config import EnrichmentConfig
from .enricher import Enricher
from .exceptions import (
    EnrichmentError,
    FieldValidationError,
    PipelineError,
    RowError,
    StepError,
)
from .hooks import (
    EnrichmentHooks,
    PipelineEndEvent,
    PipelineStartEvent,
    RowCompleteEvent,
    StepEndEvent,
    StepStartEvent,
)
from .runlog import SCHEMA_VERSION as RUN_LOG_SCHEMA_VERSION
from .runlog import JsonlRunLogger

__all__ = [
    "CacheManager",
    "Enricher",
    "CheckpointManager",
    "CheckpointData",
    "EnrichmentConfig",
    "EnrichmentError",
    "FieldValidationError",
    "StepError",
    "PipelineError",
    "RowError",
    "EnrichmentHooks",
    "PipelineStartEvent",
    "PipelineEndEvent",
    "StepStartEvent",
    "StepEndEvent",
    "RowCompleteEvent",
    "JsonlRunLogger",
    "RUN_LOG_SCHEMA_VERSION",
]
