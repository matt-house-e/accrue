"""Base schema types for the Accrue pipeline."""

from pydantic import BaseModel


class UsageInfo(BaseModel):
    """Token usage from a single LLM call.

    Attributes:
        prompt_tokens: Number of tokens in the prompt/input.
        completion_tokens: Number of tokens in the completion/output.
        total_tokens: Sum of prompt and completion tokens.
        model: Model identifier that served the request.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class StepUsage(BaseModel):
    """Aggregated token usage for a single pipeline step.

    Attributes:
        prompt_tokens: Total prompt tokens across all rows in this step.
        completion_tokens: Total completion tokens across all rows.
        total_tokens: Sum of prompt and completion tokens.
        rows_processed: Number of rows executed (cache hits + misses).
        rows_skipped: Rows skipped by ``run_if``/``skip_if`` predicates.
        model: Model identifier used by this step.
        cache_hits: Rows served from the SQLite cache.
        cache_misses: Rows that required a fresh LLM/function call.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    rows_processed: int = 0
    rows_skipped: int = 0
    model: str = ""
    cache_hits: int = 0
    cache_misses: int = 0
    execution_mode: str = "realtime"
    batch_id: str | None = None

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class CostSummary(BaseModel):
    """Aggregated cost/usage across all pipeline steps.

    Available as ``PipelineResult.cost`` after a pipeline run.

    Attributes:
        total_prompt_tokens: Sum of prompt tokens across all steps.
        total_completion_tokens: Sum of completion tokens across all steps.
        total_tokens: Sum of all tokens across all steps.
        steps: Per-step usage breakdown keyed by step name.
    """

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    steps: dict[str, StepUsage] = {}
