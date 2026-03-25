"""Convenience re-export of LLM provider adapters.

Usage:
    from accrue.providers import OpenAIClient
    from accrue.providers import AnthropicClient  # requires: pip install accrue[anthropic]
    from accrue.providers import GoogleClient      # requires: pip install accrue[google]
"""

# Optional providers — import errors deferred to instantiation
from .steps.providers.anthropic import AnthropicClient
from .steps.providers.base import LLMAPIError, LLMClient, LLMResponse
from .steps.providers.google import GoogleClient
from .steps.providers.openai import OpenAIClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMAPIError",
    "OpenAIClient",
    "AnthropicClient",
    "GoogleClient",
]
