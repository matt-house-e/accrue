# Installation

Accrue requires **Python 3.10+**.

## Install

```bash
pip install accrue                # Base install (includes OpenAI support)
pip install accrue[anthropic]     # Add Anthropic (Claude) support
pip install accrue[google]        # Add Google (Gemini) support
```

## API Keys

Set the environment variable for your provider:

```bash
export OPENAI_API_KEY="sk-..."        # Required for OpenAI (default provider)
export ANTHROPIC_API_KEY="sk-ant-..."  # Required for Anthropic
export GOOGLE_API_KEY="AI..."          # Required for Google
```

Accrue also reads from `.env` files automatically via `python-dotenv`.

## Verify

```bash
python -c "import accrue; print(accrue.__version__)"
# 1.0.0
```
