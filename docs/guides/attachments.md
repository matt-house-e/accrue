# Attachments

`LLMStep(attachments="column")` sends a row's documents (PDFs today) to the model alongside the prompt, instead of asking you to extract their text first. The model reads the file; accrue just carries the bytes.

```python
from accrue import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("read_contract",
        fields={
            "counterparty": "Name of the counterparty",
            "term_months": {"prompt": "Contract term in months", "type": "Number"},
        },
        model="claude-sonnet-4-5-20250929",
        attachments="documents",
    ),
])

result = pipeline.run([
    {"deal": "Acme", "documents": "contracts/acme.pdf"},
    {"deal": "Beta", "documents": ["contracts/beta.pdf", "contracts/beta-amendment.pdf"]},
])
```

The named column is removed from the `<row_data>` JSON in the prompt and rendered as provider document blocks ahead of the user text. Every other column still reaches the prompt as usual.

## What a cell can hold

| Cell value | Becomes |
|---|---|
| `"path/to/file.pdf"` or a `pathlib.Path` | one `Document` read from disk |
| `b"%PDF-..."` | one `Document` with `media_type="application/pdf"` |
| `Document(...)` | itself |
| a `list`/`tuple`/numpy array of any of the above | one `Document` each, in order |
| `None`, `NaN`, `pd.NA`, `""`, `[]` | no attachments — a plain text prompt |

Anything else raises `StepError` naming the offending type, as does a path that cannot be read. A row that lacks the column altogether also raises, so a misspelt column name fails every row before any call instead of quietly sending prompts without their documents. On the batch path these are per-row errors raised before anything is submitted; the rest of the batch still goes out.

## `Document`

```python
from accrue import Document

doc = Document.from_path("report.pdf")              # title defaults to "report.pdf"
doc = Document.from_path("report.pdf", title="Q3")  # media type guessed via mimetypes
doc = Document(data=pdf_bytes, media_type="application/pdf", title="Q3")

doc.sha256   # hex digest of the bytes — the cache identity
```

Frozen dataclass, and its `repr` prints the title, media type, and byte count — never the bytes.

## Anthropic only

Attachments go out as Anthropic `document` blocks with a base64 source. `OpenAIClient` and `GoogleClient` raise `StepError` before any network call if a step hands them document blocks. Use a `claude-*` model (the provider is auto-detected from the model prefix).

## Caching

The cache key hashes the document **bytes**, not the path. The same PDF under two filenames is one cache entry; editing a PDF in place invalidates the entry for that row.

## Run logs

At `capture="prompts"` the prompt sidecar stores a placeholder instead of the file:

```json
{"type": "document", "title": "acme.pdf", "media_type": "application/pdf",
 "bytes": 184320, "sha256": "9f2c..."}
```

Enough to tell two attachments apart, without putting a megabyte of base64 into every run log.

## Request limits

Anthropic caps a single request at **32 MB** — measured after base64 encoding, which adds a third, so roughly 24 MB of raw PDF — and at **100 PDF pages** (600 on 1M-context models). The Message Batches API caps a batch at **256 MB** total, so a large PDF set has to be split, for example with a small `EnrichmentConfig(batch_max_requests=...)` — accrue does not size batches by attachment bytes automatically.

A response cut off at `max_tokens` fails the row with an error naming the limit rather than being retried: every retry would resend the full request, documents included, and be cut off again. Raise `max_tokens` (above roughly 21,000 also pass `AnthropicClient(timeout=...)`, see [providers](providers.md)).

Attachment bytes are input tokens: a long PDF is a real cost, and the cached system prefix does not cover them.

## Out of scope

- Text extraction, OCR, or chunking — send the file, or pre-process it yourself in a `FunctionStep`.
- Images, audio, and video blocks.
- Remote URLs, the Anthropic Files API, and provider-side document caching.
- Per-attachment token accounting beyond what the provider reports.
