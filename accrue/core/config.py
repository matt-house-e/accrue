"""
Unified configuration for the Accrue enrichment tool.

Defaults are tuned for fast processing on OpenAI Tier 2+ accounts.
For Tier 1 accounts, reduce max_workers to 5-10.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnrichmentConfig:
    """
    Configuration for enrichment pipelines.

    Defaults are optimized for speed on typical API tier accounts.
    Adjust max_workers based on your provider's rate limits.

    Rate limit guidance (OpenAI GPT-4.1 nano/mini):
      - Tier 1 ($5 spend):   max_workers=5-10   (500 RPM, 200K TPM)
      - Tier 2 ($50 spend):  max_workers=20-50   (5K RPM, 2M TPM)
      - Tier 3+ ($100+):     max_workers=50-100  (5K+ RPM, 4M+ TPM)
      - Tier 5 ($1K+):       max_workers=100-200 (30K RPM, 150M TPM)
    """

    # === LLM Configuration ===
    max_tokens: int = 4000
    """Maximum tokens for LLM output."""

    temperature: float = 0.2
    """LLM temperature. Low (0.1-0.3) is best for structured enrichment."""

    # === Concurrency ===
    max_workers: int = 10
    """Concurrent rows per step (asyncio.Semaphore bound).
    Real-world production uses 20-30. Default 10 is safe for Tier 1-2."""

    # === Fields ===
    overwrite_fields: bool = False
    """Whether to overwrite existing field values in the DataFrame."""

    # === Reliability ===
    max_retries: int = 3
    """Maximum retry attempts for API errors (429, 500, timeouts)."""

    retry_base_delay: float = 1.0
    """Base delay for exponential backoff on API errors (seconds).
    Actual delay: base_delay * 2^attempt + jitter."""

    on_error: str = "continue"
    """Error handling mode: 'continue' collects errors and returns partial results,
    'raise' fails fast on the first row error."""

    # === Logging ===
    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    enable_progress_bar: bool = True
    """Show tqdm progress bar during pipeline execution."""

    # === Checkpointing ===
    enable_checkpointing: bool = False
    """Save results after each step completes for crash recovery."""

    checkpoint_dir: str | None = None
    """Directory for checkpoint files. None = temp directory."""

    auto_resume: bool = True
    """Automatically resume from checkpoint on re-run."""

    # === Caching ===
    enable_caching: bool = False
    """Enable input-hash cache to skip redundant API calls."""

    cache_ttl: int = 3600
    """Cache time-to-live in seconds."""

    cache_dir: str = ".accrue"
    """Directory for cache.db."""

    checkpoint_interval: int = 0
    """Save partial step progress every N rows. 0 = disabled."""

    # === Batch API ===
    batch_poll_interval: float = 60.0
    """Seconds between batch job status checks."""

    batch_timeout: float = 86400.0
    """Maximum seconds to wait for a batch job to complete (default: 24h)."""

    batch_max_requests: int = 50000
    """Maximum requests per batch submission. Larger sets are auto-chunked."""

    # === Validation ===
    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.cache_ttl < 0:
            raise ValueError(f"cache_ttl must be non-negative, got {self.cache_ttl}")
        if self.checkpoint_interval < 0:
            raise ValueError(
                f"checkpoint_interval must be non-negative, got {self.checkpoint_interval}"
            )
        if self.batch_poll_interval <= 0:
            raise ValueError(
                f"batch_poll_interval must be positive, got {self.batch_poll_interval}"
            )
        if self.batch_timeout <= 0:
            raise ValueError(f"batch_timeout must be positive, got {self.batch_timeout}")
        if self.batch_max_requests <= 0:
            raise ValueError(f"batch_max_requests must be positive, got {self.batch_max_requests}")

    # === Presets ===
    @classmethod
    def for_development(cls) -> "EnrichmentConfig":
        """Low concurrency, verbose logging, caching on. Safe for Tier 1 accounts."""
        return cls(
            max_workers=5,
            temperature=0.2,
            enable_progress_bar=True,
            log_level="DEBUG",
            enable_caching=True,
        )

    @classmethod
    def for_production(cls) -> "EnrichmentConfig":
        """High concurrency with caching and checkpointing. For Tier 2+ accounts."""
        return cls(
            max_workers=30,
            temperature=0.2,
            enable_checkpointing=True,
            enable_caching=True,
            checkpoint_interval=100,
            max_retries=5,
            log_level="INFO",
        )

    @classmethod
    def for_server(cls) -> "EnrichmentConfig":
        """Async server context (FastAPI). No progress bars, high concurrency."""
        return cls(
            max_workers=30,
            temperature=0.2,
            enable_progress_bar=False,
            max_retries=5,
            log_level="WARNING",
        )

    @classmethod
    def for_batch(cls) -> "EnrichmentConfig":
        """Batch API execution with caching. For cost-optimised large datasets."""
        return cls(
            max_workers=10,
            temperature=0.2,
            enable_caching=True,
            enable_checkpointing=True,
            batch_poll_interval=60.0,
            batch_timeout=86400.0,
            max_retries=5,
            log_level="INFO",
        )
