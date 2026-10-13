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
| a `list`/`tuple` of any of the above | one `Document` each, in order |
| `None`, `NaN`, `pd.NA`, `""`, `[]` | no attachments — a plain text prompt |

Anything else raises `StepError` naming the offending type, as does a path that cannot be read.

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

Providers cap request size, and attachments are where you hit the caps. Base64 encoding inflates every file by 4/3, so 25 MB of PDFs is about 34 MB on the wire.

| Limit | Anthropic | OpenAI |
|---|---|---|
| One request | 32 MB | none documented, not checked |
| One batch submission | 256 MB (and 100,000 requests) | 200 MB input file (and 50,000 requests) |
| PDF pages per request | 600, or 100 under a 1M-token context window | n/a (attachments are Anthropic-only) |

Sources: Anthropic [request size limits](https://platform.claude.com/docs/en/api/overview#request-size-limits), [batch limitations](https://platform.claude.com/docs/en/build-with-claude/batch-processing#batch-limitations), [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support); OpenAI [Batch API](https://developers.openai.com/api/docs/guides/batch). accrue reads MB as 1,000,000 bytes, the conservative reading.

What accrue does about them:

- **It estimates each request's size before sending.** Document bytes are counted at their base64 length without being encoded; the rest of the request (prompt, schema, tools) is measured by serializing it, plus 1 KB for the provider envelope.
- **An over-limit row fails alone.** If a row's request exceeds the client's per-request limit, the row gets a `StepError` naming the estimated size and the limit, no API call is made, and the rest of the run continues. This applies to realtime and batch steps. `Pipeline.retry_failed()` will retry the row, so shrink or split its attachments first.
- **Batches are split by size as well as count.** A new batch starts before the summed estimate would pass `EnrichmentConfig.batch_max_bytes`, which defaults to the client's per-batch cap. A request that is bigger than the batch limit on its own fails that row instead of the batch.

The limits are the client attributes `max_request_bytes` and `max_batch_bytes`, in bytes. Set them on an instance for a gateway with tighter caps (Bedrock allows 20 MB per request, Google Cloud 30 MB), or to `None` to switch a check off:

```python
client = AnthropicClient()
client.max_request_bytes = 20_000_000

config = EnrichmentConfig(batch_max_bytes=100_000_000)  # smaller batches
```

A custom client without these attributes is not checked.

**Pages and tokens are not checked.** Counting PDF pages reliably means parsing the PDF (page trees, compressed object streams, incremental updates), and accrue does not take a PDF dependency for that; a regex over the raw bytes miscounts real files. Token counts depend on the model's tokenizer and on how the provider renders each page. A file over the page limit, or a prompt over the context window, is still refused by the provider: on a realtime step the row fails with the provider's error, and on a batch step that one request fails inside the batch and is retried realtime, where it fails the same way. If your files are near the page limit, count pages yourself (for example with `pypdf` in a `FunctionStep`) and skip or split those rows before the `LLMStep`.

Attachment bytes are input tokens: a long PDF is a real cost, and the cached system prefix does not cover them.

## Out of scope

- Text extraction, OCR, or chunking — send the file, or pre-process it yourself in a `FunctionStep`.
- Images, audio, and video blocks.
- Remote URLs, the Anthropic Files API, and provider-side document caching.
- Per-attachment token accounting beyond what the provider reports.
