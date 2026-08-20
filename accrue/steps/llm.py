"""LLMStep — calls an LLM provider for structured enrichment."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from ..core.exceptions import PipelineError, StepError
from ..core.hooks import RowAttemptEvent
from ..schemas.enrichment import EnrichmentResult
from ..schemas.field_spec import FieldSpec
from ..schemas.grounding import GroundingConfig
from ..utils.logger import get_logger
from .base import StepContext, StepResult
from .prompt_builder import PromptParts, build_prompt
from .providers.base import LLMAPIError, LLMClient, LLMResponse, is_batch_capable
from .providers.openai import OpenAIClient
from .schema_builder import build_json_schema, build_response_model

# Optional provider imports for structured output auto-detection
try:
    from .providers.anthropic import AnthropicClient as _AnthropicClient
except ImportError:
    _AnthropicClient = None

try:
    from .providers.google import GoogleClient as _GoogleClient
except ImportError:
    _GoogleClient = None

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Refusal detection for default enforcement
# ---------------------------------------------------------------------------
def _strip_markdown_fences(content: str) -> str:
    """Strip surrounding markdown code fences if present.

    Some Claude models (notably Haiku with grounding tools) wrap JSON
    in ``` ... ``` fences when not constrained by structured outputs.
    Safe to call on unfenced content — returns it unchanged.
    """
    s = content.strip()
    if not s.startswith("```"):
        return content
    lines = s.split("\n")
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


REFUSAL_PATTERNS = frozenset(
    {
        "unable to determine",
        "n/a",
        "not available",
        "not specified",
        "insufficient data",
        "unknown",
        "not enough information",
        "cannot determine",
        "no data",
        "no information",
        "not applicable",
        "data not available",
        "information not available",
    }
)


class LLMStep:
    """Calls an LLM provider to produce enrichment values.

    Features:
      - Provider-agnostic via LLMClient protocol (OpenAI default).
      - Lazy client initialisation (no import-time API key check).
      - 7-key field spec validation via :class:`FieldSpec` on construction.
      - Dynamic system prompt (markdown headers + XML data boundaries).
      - Default enforcement: replaces LLM refusals with field ``default`` values.
      - Uses ``response_format={"type": "json_object"}``.
      - Validates response with Pydantic ``model_validate()``.
      - On validation/parse error: appends error to conversation and retries.
    """

    def __init__(
        self,
        name: str,
        fields: list[str] | dict[str, str | dict],
        depends_on: list[str] | None = None,
        model: str = "gpt-4.1-mini",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        system_prompt_header: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: LLMClient | None = None,
        schema: Type[BaseModel] = EnrichmentResult,
        max_retries: int = 2,
        cache: bool = True,
        structured_outputs: bool | None = None,
        grounding: bool | dict | GroundingConfig | None = None,
        sources_field: str | None = "sources",
        run_if: Callable[..., Any] | None = None,
        skip_if: Callable[..., Any] | None = None,
        batch: bool = False,
        provider_kwargs: dict[str, Any] | None = None,
    ):
        """Configure an LLM enrichment step.

        Args:
            name: Unique step name used in logs, cache keys, and ``depends_on``
                references.
            fields: Fields this step produces.  Two forms:

                * ``list[str]`` — field names only (prompts come from CSV or
                  external field specs).
                * ``dict[str, str | dict]`` — inline field specs.  String
                  values are shorthand for ``{"prompt": value}``.  Dict values
                  are validated as :class:`FieldSpec` (keys: ``prompt``,
                  ``type``, ``format``, ``enum``, ``examples``,
                  ``bad_examples``, ``default``).

            depends_on: Names of steps whose outputs this step needs.  The
                pipeline resolves these as DAG edges.
            model: Model identifier passed to the LLM provider.
            temperature: Sampling temperature.  Falls back to
                ``config.temperature`` then ``0.2``.
            max_tokens: Maximum response tokens.  Falls back to
                ``config.max_tokens`` then ``4000``.
            system_prompt: **Tier 3** — fully replaces the auto-generated
                system prompt.  Use only when the dynamic prompt builder
                doesn't fit your needs.
            system_prompt_header: **Tier 2** — injected as a ``# Context``
                section between the Role header and the Field Specification
                Keys.  Ignored when ``system_prompt`` is set.
            api_key: Provider API key.  Falls back to the relevant env var
                (e.g. ``OPENAI_API_KEY``).
            base_url: OpenAI-compatible base URL (Ollama, Groq, etc.).
                Disables structured-output auto-detection.
            client: Pre-configured :class:`LLMClient` instance.  Overrides
                ``api_key`` and ``base_url``.
            schema: Pydantic model for response validation.  Default
                ``EnrichmentResult`` works with dynamic field specs.
            max_retries: Parse/validation retry attempts per API call.
            cache: Enable input-hash caching for this step (default True).
            structured_outputs: Override structured-output auto-detection.
                ``True`` forces ``json_schema``; ``False`` forces
                ``json_object``; ``None`` (default) auto-detects based on
                provider and field specs.
            grounding: Enable provider-level web search grounding.
                ``True`` enables with defaults, a ``dict`` or
                :class:`GroundingConfig` allows fine-grained control
                (``allowed_domains``, ``blocked_domains``, ``user_location``,
                ``max_searches``, ``provider_kwargs``).  ``None`` or ``False``
                disables.  Use ``provider_kwargs`` to pass provider-specific
                options (e.g. ``{"search_context_size": "high"}`` for OpenAI).
            sources_field: Name of the visible output field for grounding
                citations.  Defaults to ``"sources"``.  Set to ``None`` to
                disable citation injection entirely.  Silently ignored when
                grounding is disabled.  Not included in the cache key
                (changing it does not invalidate cached results).
            run_if: Predicate ``(row, prior_results) -> bool``.  When set,
                the step only runs for rows where the predicate returns True.
                Mutually exclusive with ``skip_if``.
            skip_if: Predicate ``(row, prior_results) -> bool``.  When set,
                the step is skipped for rows where the predicate returns True.
                Mutually exclusive with ``run_if``.
            batch: Use provider Batch API for this step.  Requires the
                configured client to implement
                :class:`BatchCapableLLMClient`.  Mutually exclusive with
                ``grounding`` (batch APIs do not support tool use).
            provider_kwargs: Extra keyword arguments merged into the
                provider's API call.  Use this as an escape hatch for
                provider-specific features not yet exposed as first-class
                parameters (e.g.
                ``{"thinking": {"type": "adaptive"}}`` for Anthropic
                extended thinking, or ``{"effort": "high"}``).
                These are **not** included in the cache key.
        """
        if run_if is not None and skip_if is not None:
            raise PipelineError(
                f"Step '{name}' has both run_if and skip_if set. "
                f"These are mutually exclusive — use one or the other."
            )
        self.name = name
        self.depends_on = depends_on or []
        self.model = model
        self.temperature = temperature
        self.cache = cache
        self.batch = batch
        self.max_tokens = max_tokens
        self._custom_system_prompt = system_prompt
        self._system_prompt_header = system_prompt_header
        self.api_key = api_key
        self.base_url = base_url
        self.schema = schema
        self.max_retries = max_retries
        self._client: LLMClient | None = client
        self._structured_outputs_param = structured_outputs
        self.run_if = run_if
        self.skip_if = skip_if
        self.provider_kwargs = provider_kwargs
        # Warn-once guard: is_batch_eligible is a property and may be read more
        # than once per run (pipeline dispatch, plan(), user code).
        self._warned_batch_unavailable: bool = False

        # Normalize grounding config: True → GroundingConfig(), dict → validated
        self._grounding_config: GroundingConfig | None = _normalize_grounding(grounding)
        self.sources_field = sources_field

        # Validate batch + grounding conflict
        if batch and self._grounding_config is not None:
            raise PipelineError(
                f"Step '{name}' has both batch=True and grounding enabled. "
                f"Batch APIs do not support tool use (web search). "
                f"Disable grounding or set batch=False."
            )

        # Normalize fields: dict → inline FieldSpec objects + field names list
        if isinstance(fields, dict):
            self._field_specs = self._normalize_field_specs(fields)
            self.fields = list(fields.keys())
        else:
            self._field_specs: dict[str, FieldSpec] = {}
            self.fields = fields

        # Validate sources_field doesn't conflict with declared fields
        if (
            self._grounding_config is not None
            and self.sources_field is not None
            and self.sources_field in self.fields
        ):
            raise PipelineError(
                f"Step '{name}': sources_field '{self.sources_field}' conflicts "
                f"with a declared field name. Use sources_field=None to disable "
                f"citation injection, or choose a different name."
            )

        # Build and cache structured outputs format (field specs are immutable)
        self._response_format = self._build_response_format()
        self._use_structured_outputs = self._response_format.get("type") == "json_schema"

    @staticmethod
    def _normalize_field_specs(fields: dict[str, str | dict]) -> dict[str, FieldSpec]:
        """Convert shorthand field specs to validated FieldSpec objects.

        ``{"market_size": "Estimate TAM"}`` → ``{"market_size": FieldSpec(prompt="Estimate TAM")}``
        ``{"market_size": {"prompt": "...", "type": "String"}}`` → validated FieldSpec
        """
        result: dict[str, FieldSpec] = {}
        for name, spec in fields.items():
            if isinstance(spec, str):
                result[name] = FieldSpec(prompt=spec)
            else:
                result[name] = FieldSpec.model_validate(spec)
        return result

    # -- client ----------------------------------------------------------

    def _resolve_client(self) -> LLMClient:
        """Lazily create or return the LLMClient.

        Picks the provider by model-name prefix when no explicit ``client``
        was passed:

          * ``claude-*``  → ``AnthropicClient``  (requires ``accrue[anthropic]``)
          * ``gemini-*``  → ``GoogleClient``    (requires ``accrue[google]``)
          * everything else, or any model when ``base_url`` is set → ``OpenAIClient``

        ``base_url`` always implies an OpenAI-compatible endpoint
        (Ollama, Groq, etc.) regardless of model name; users running
        Claude or Gemini through a non-OpenAI gateway should pass
        ``client=`` explicitly.
        """
        if self._client is not None:
            return self._client

        if self.base_url is None and self.model.startswith("claude-"):
            if _AnthropicClient is None:
                raise ImportError(
                    f"Model '{self.model}' requires the Anthropic provider. "
                    "Install with: pip install accrue[anthropic]"
                )
            self._client = _AnthropicClient(api_key=self.api_key)
        elif self.base_url is None and self.model.startswith("gemini-"):
            if _GoogleClient is None:
                raise ImportError(
                    f"Model '{self.model}' requires the Google provider. "
                    "Install with: pip install accrue[google]"
                )
            self._client = _GoogleClient(api_key=self.api_key)
        else:
            self._client = OpenAIClient(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    # -- response format -------------------------------------------------

    def _build_response_format(self) -> dict:
        """Determine the response_format based on auto-detection or explicit override.

        Auto-detect logic:
          - Custom schema → json_object (user manages validation)
          - No field specs (list fields) → json_object
          - Non-OpenAI client → json_object
          - OpenAI with base_url and structured_outputs is None → json_object
          - OpenAI native (no base_url) → json_schema
          - structured_outputs=True → force json_schema
          - structured_outputs=False → force json_object
        """
        # Explicit override
        if self._structured_outputs_param is False:
            return {"type": "json_object"}

        if self._structured_outputs_param is True:
            # Force on — requires field specs
            if self._field_specs:
                return build_json_schema(self._field_specs)
            return {"type": "json_object"}

        # Auto-detect (structured_outputs is None)

        # Custom schema → json_object
        if self.schema is not EnrichmentResult:
            return {"type": "json_object"}

        # No field specs (list[str] fields) → json_object
        if not self._field_specs:
            return {"type": "json_object"}

        # Known providers that support structured outputs (json_schema)
        if self._client is not None and not isinstance(self._client, OpenAIClient):
            _supports_structured = (
                _AnthropicClient is not None and isinstance(self._client, _AnthropicClient)
            ) or (_GoogleClient is not None and isinstance(self._client, _GoogleClient))

            if _supports_structured:
                return build_json_schema(self._field_specs)

            # Unknown custom client → json_object (safe fallback)
            return {"type": "json_object"}

        # OpenAI with base_url → json_object (third-party compatibility)
        if self.base_url is not None:
            return {"type": "json_object"}

        # Native OpenAI → json_schema
        return build_json_schema(self._field_specs)

    # -- message building ------------------------------------------------

    def _build_prompt(self, ctx: StepContext) -> PromptParts:
        """Build the system/user prompt halves using the dynamic prompt builder.

        The ``system`` half is identical for every row of this step, which is
        what makes provider prompt caching actually hit — see
        :mod:`accrue.steps.prompt_builder`.
        """
        return build_prompt(
            field_specs=self._field_specs,
            row=ctx.row,
            prior_results=ctx.prior_results or None,
            custom_system_prompt=self._custom_system_prompt,
            system_prompt_header=self._system_prompt_header,
        )

    # -- tools -----------------------------------------------------------

    def _build_tools_config(self) -> list[dict[str, Any]] | None:
        """Build the tools list for the LLM client when grounding is enabled."""
        if self._grounding_config is None:
            return None
        tool: dict[str, Any] = {"type": "web_search"}
        cfg = self._grounding_config
        if cfg.allowed_domains:
            tool["allowed_domains"] = cfg.allowed_domains
        if cfg.blocked_domains:
            tool["blocked_domains"] = cfg.blocked_domains
        if cfg.user_location:
            tool["user_location"] = cfg.user_location
        if cfg.max_searches is not None:
            tool["max_searches"] = cfg.max_searches
        if cfg.provider_kwargs:
            tool["provider_kwargs"] = cfg.provider_kwargs
        return [tool]

    # -- default enforcement ---------------------------------------------

    def _apply_defaults(self, values: dict[str, Any]) -> dict[str, Any]:
        """Replace refusal values with field defaults where configured.

        Skip the override when the field declares an ``enum`` and the value
        (case-insensitive, stripped) is a member of that enum — the model's
        output is then a legitimate categorical answer, not a refusal.
        """
        for field_name, spec in self._field_specs.items():
            if field_name not in values:
                continue
            if "default" not in spec.model_fields_set:
                continue
            if not _is_refusal(values[field_name]):
                continue
            # Value looks like a refusal — check whether it is a valid enum member
            if spec.enum is not None and isinstance(values[field_name], str):
                normalized = values[field_name].strip().lower()
                enum_lower = {v.lower() for v in spec.enum}
                if normalized in enum_lower:
                    logger.debug(
                        "LLMStep._apply_defaults: field '%s' value %r matches "
                        "REFUSAL_PATTERNS but is a declared enum member — keeping.",
                        field_name,
                        values[field_name],
                    )
                    continue
            logger.debug(
                "LLMStep._apply_defaults: field '%s' value %r matches "
                "REFUSAL_PATTERNS — replacing with default %r.",
                field_name,
                values[field_name],
                spec.default,
            )
            values[field_name] = spec.default
        return values

    # -- batch helpers ---------------------------------------------------

    @property
    def is_batch_eligible(self) -> bool:
        """True when this step should use the batch execution path.

        Requires ``batch=True`` and a client that is batch-capable *in its
        current configuration* — see
        :func:`~accrue.steps.providers.base.is_batch_capable`.  A client whose
        class implements the batch methods is not enough: ``OpenAIClient``
        pointed at a gateway via ``base_url`` has them but no endpoint behind
        them.

        When ``batch=True`` and the client cannot batch, the step degrades to
        realtime execution and warns once.  Silently paying realtime prices for
        a run the caller asked to batch is an expensive surprise.
        """
        if not self.batch:
            return False
        client = self._resolve_client()
        if is_batch_capable(client):
            return True
        self._warn_batch_unavailable(client)
        return False

    def _warn_batch_unavailable(self, client: LLMClient) -> None:
        """Warn once that ``batch=True`` is being downgraded to realtime.

        Args:
            client: The resolved client that cannot run batch jobs.
        """
        if self._warned_batch_unavailable:
            return
        self._warned_batch_unavailable = True
        logger.warning(
            "Step '%s' requested batch=True but %s cannot run batch jobs in this "
            "configuration; falling back to realtime execution at standard "
            "(non-batch) pricing.",
            self.name,
            type(client).__name__,
        )

    def build_messages(self, ctx: StepContext) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Build messages and call kwargs for a single row.

        Used by both the realtime path (``run()``) and the batch execution
        path in ``Pipeline._execute_step_batch()``.

        Args:
            ctx: Step context containing row data, field specs, and config.

        Returns:
            Tuple of ``(messages, call_kwargs)`` where *call_kwargs* contains
            ``model``, ``temperature``, ``max_tokens``, ``response_format``,
            and ``tools``.
        """
        temperature = self.temperature
        if temperature is None and ctx.config is not None:
            temperature = ctx.config.temperature
        if temperature is None:
            temperature = 0.2

        max_tokens = self.max_tokens
        if max_tokens is None and ctx.config is not None:
            max_tokens = ctx.config.max_tokens
        if max_tokens is None:
            max_tokens = 4000

        prompt = self._build_prompt(ctx)
        tools = self._build_tools_config()

        # The system message holds only step-static content so providers can
        # cache it across rows; row data rides in the user message.
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": self._response_format,
            "tools": tools,
            "provider_kwargs": self.provider_kwargs,
        }

        return messages, call_kwargs

    def parse_response(self, response: LLMResponse) -> StepResult:
        """Parse and validate an LLM response into a StepResult.

        Performs JSON decoding, Pydantic validation, field filtering, default
        enforcement, and citation injection.  Used by both the realtime
        ``run()`` path and the batch execution path.

        Args:
            response: Raw LLM response from a provider.

        Returns:
            Validated step result.

        Raises:
            json.JSONDecodeError: If the response is not valid JSON.
            pydantic.ValidationError: If the parsed JSON fails schema
                validation.
        """
        content = response.content
        parsed = json.loads(_strip_markdown_fences(content))

        # Extract __ internal fields before Pydantic validation (which
        # rejects them due to extra="forbid").  They bypass schema
        # validation but are still passed between steps.
        internal_values = {k: v for k, v in parsed.items() if k.startswith("__")}
        parsed_clean = {k: v for k, v in parsed.items() if not k.startswith("__")}

        if self._use_structured_outputs:
            dynamic_model = build_response_model(self._field_specs)
            validated = dynamic_model.model_validate(parsed_clean)
        else:
            validated = self.schema.model_validate(parsed_clean)
        all_values = validated.model_dump()
        all_values.update(internal_values)

        values = {k: v for k, v in all_values.items() if k in self.fields}
        values = self._apply_defaults(values)

        if response.citations and self.sources_field is not None:
            values[self.sources_field] = [
                {"url": c.url, "title": c.title, "snippet": c.snippet} for c in response.citations
            ]

        return StepResult(
            values=values,
            usage=response.usage,
            metadata={
                "raw_response": content,
                "structured_outputs": self._use_structured_outputs,
            },
        )

    # -- run -------------------------------------------------------------

    async def run(self, ctx: StepContext) -> StepResult:
        """Execute the LLM call with two-layer retry.

        Outer loop: API errors (429, 500, timeouts) with exponential backoff.
            Uses config.max_retries and config.retry_base_delay.
        Inner loop: Parse/validation errors fed back to the LLM.
            Uses self.max_retries (step-level).

        One :class:`~accrue.core.hooks.RowAttemptEvent` is fired per provider
        call (#134): ``kind="api"`` when the call itself raised, ``kind="parse"``
        when it returned and we tried to parse it (``status="ok"`` on success).
        At ``ctx.capture >= "prompts"`` the event also carries the rendered
        request/response so the run logger can persist it to the prompt sidecar.
        """
        client = self._resolve_client()
        messages, call_kwargs = self.build_messages(ctx)
        temperature = call_kwargs["temperature"]
        max_tokens = call_kwargs["max_tokens"]
        tools = call_kwargs["tools"]

        # API retry config from EnrichmentConfig
        api_max_retries = 3
        retry_base_delay = 1.0
        if ctx.config is not None:
            api_max_retries = ctx.config.max_retries
            retry_base_delay = ctx.config.retry_base_delay

        # Attempt-event plumbing (#134).  When nothing is listening
        # (``on_attempt is None``) every branch below short-circuits, so a
        # metadata run over a non-logged pipeline pays nothing.
        on_attempt = ctx.on_attempt
        row_index = ctx.row_index if ctx.row_index is not None else -1
        capture_bodies = on_attempt is not None and ctx.capture in ("prompts", "full")

        last_api_error: BaseException | None = None
        total_attempts = 0

        async def _emit(
            kind: str,
            status: str,
            latency_ms: float,
            backoff_s: float | None,
            error: BaseException | None,
            body: dict[str, Any] | None,
        ) -> None:
            if on_attempt is None:
                return
            await on_attempt(
                RowAttemptEvent(
                    step_name=self.name,
                    row_index=row_index,
                    attempt=total_attempts,
                    kind=kind,
                    status=status,
                    latency_ms=latency_ms,
                    backoff_s=backoff_s,
                    error=error,
                    body=body,
                )
            )

        for api_attempt in range(api_max_retries + 1):
            # Reset messages for each API retry (parse retries accumulate within)
            if api_attempt > 0:
                messages, _ = self.build_messages(ctx)

            last_parse_error: BaseException | None = None
            content: str = ""
            call_t0 = time.monotonic()

            try:
                # An LLMAPIError from complete() is not caught here; it falls
                # through to the outer handler (the API-retry loop).
                for parse_attempt in range(self.max_retries + 1):
                    total_attempts += 1
                    call_t0 = time.monotonic()
                    try:
                        response: LLMResponse = await client.complete(
                            messages=messages,
                            model=self.model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=self._response_format,
                            tools=tools,
                            provider_kwargs=self.provider_kwargs,
                        )
                    except TypeError:
                        if tools is not None:
                            raise StepError(
                                f"LLMStep '{self.name}' uses grounding but the "
                                f"configured LLM client does not support the 'tools' "
                                f"parameter.  Use a built-in provider adapter "
                                f"(OpenAIClient, AnthropicClient, GoogleClient) or "
                                f"update your custom client's complete() signature.",
                                step_name=self.name,
                            )
                        raise

                    try:
                        result = self.parse_response(response)
                    except (json.JSONDecodeError, ValidationError) as exc:
                        latency_ms = (time.monotonic() - call_t0) * 1000.0
                        last_parse_error = exc
                        content = getattr(response, "content", "")
                        status = (
                            "parse_error"
                            if isinstance(exc, json.JSONDecodeError)
                            else "validation_error"
                        )
                        await _emit(
                            "parse",
                            status,
                            latency_ms,
                            None,
                            exc,
                            _capture_body(messages, response, None) if capture_bodies else None,
                        )
                        logger.warning(
                            "LLMStep '%s' parse attempt %d failed: %s",
                            self.name,
                            parse_attempt + 1,
                            exc,
                        )
                        # Feed the error back so the LLM can self-correct
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your response was invalid: {exc}. "
                                    "Please return valid JSON matching the required schema."
                                ),
                            },
                        )
                        continue

                    latency_ms = (time.monotonic() - call_t0) * 1000.0
                    await _emit(
                        "parse",
                        "ok",
                        latency_ms,
                        None,
                        None,
                        _capture_body(messages, response, result.values)
                        if capture_bodies
                        else None,
                    )
                    # Enrich metadata with retry info
                    result.metadata["attempts"] = total_attempts
                    result.metadata["api_retries"] = api_attempt
                    return result

                # All parse retries exhausted
                raise StepError(
                    f"LLMStep '{self.name}' failed after {self.max_retries + 1} "
                    f"parse attempts (model={self.model}): {last_parse_error}. "
                    f"Check that the model supports JSON output and that field "
                    f"specs are unambiguous.",
                    step_name=self.name,
                )

            except LLMAPIError as exc:
                latency_ms = (time.monotonic() - call_t0) * 1000.0
                if not exc.retryable:
                    await _emit("api", _api_status(exc), latency_ms, None, exc, None)
                    raise
                last_api_error = exc
                backoff: float | None = None
                if api_attempt < api_max_retries:
                    # Full random jitter: uniform(0, base * 2^attempt)
                    delay = random.uniform(0, retry_base_delay * (2**api_attempt))
                    # Respect Retry-After header: use it as a floor
                    if exc.retry_after is not None:
                        delay = max(delay, float(exc.retry_after))
                    backoff = delay
                await _emit("api", _api_status(exc), latency_ms, backoff, exc, None)
                if backoff is not None:
                    logger.warning(
                        "LLMStep '%s' API error (attempt %d/%d), retrying in %.1fs: %s",
                        self.name,
                        api_attempt + 1,
                        api_max_retries + 1,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)

        raise StepError(
            f"LLMStep '{self.name}' API error after {api_max_retries + 1} retries "
            f"(model={self.model}): {last_api_error}. "
            f"Check your API key, rate limits, and model availability.",
            step_name=self.name,
        )


def _normalize_grounding(
    grounding: bool | dict | GroundingConfig | None,
) -> GroundingConfig | None:
    """Normalize the ``grounding`` constructor argument.

    ``True`` → ``GroundingConfig()``; ``dict`` → validated; ``None``/``False`` → ``None``.
    """
    if grounding is None or grounding is False:
        return None
    if grounding is True:
        return GroundingConfig()
    if isinstance(grounding, GroundingConfig):
        return grounding
    if isinstance(grounding, dict):
        return GroundingConfig.model_validate(grounding)
    raise PipelineError(
        f"Invalid grounding value: {grounding!r}. "
        f"Expected True, False, None, dict, or GroundingConfig."
    )


def _api_status(exc: LLMAPIError) -> str:
    """Short status string for a provider API error, for the attempt event (#134)."""
    if getattr(exc, "is_rate_limit", False) or exc.status_code == 429:
        return "rate_limited"
    if exc.status_code == 408:
        return "timeout"
    return "api_error"


def _capture_body(
    messages: list[dict[str, str]],
    response: LLMResponse,
    parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    """Snapshot the rendered request/response for one attempt (capture>=prompts, #134).

    ``messages`` is copied because the parse-retry loop appends to it after the
    attempt event is emitted.  Secrets are redacted downstream by the run logger
    before the body ever touches disk.

    Raw provider request/response payloads (the ``full`` tier) are not cheaply
    available from the provider adapters, which return a normalised
    :class:`LLMResponse` — so ``full`` captures the same body as ``prompts``.
    """
    return {
        "messages": [dict(m) for m in messages],
        "response": getattr(response, "content", None),
        "parsed": parsed,
    }


def _is_refusal(value: Any) -> bool:
    """Check if a value looks like an LLM refusal."""
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized == "" or normalized in REFUSAL_PATTERNS
    return False
