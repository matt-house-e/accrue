"""Per-row document attachments for :class:`~accrue.steps.llm.LLMStep` (#153).

A :class:`Document` is accrue's provider-neutral wrapper around a binary file
(a PDF today) that rides along with one row's prompt.  :func:`normalize_attachments`
turns whatever a DataFrame cell happens to hold — a path, raw bytes, a
``Document``, or a list of any of those — into ``list[Document]``.

Provider adapters are responsible for encoding; accrue carries raw bytes so the
run log can replace them with a digest instead of a megabyte of base64.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.exceptions import StepError

DEFAULT_MEDIA_TYPE = "application/pdf"


@dataclass(frozen=True)
class Document:
    """A single binary document attached to one row's LLM call.

    Attributes:
        data: Raw file bytes.  Providers encode these; accrue never does.
        media_type: IANA media type (default ``application/pdf``).
        title: Optional human-readable title passed to the provider.
    """

    data: bytes
    media_type: str = DEFAULT_MEDIA_TYPE
    title: str | None = None

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike[str],
        title: str | None = None,
        media_type: str | None = None,
    ) -> Document:
        """Read a document off disk.

        Args:
            path: File to read.
            title: Defaults to the file name.
            media_type: Defaults to a :mod:`mimetypes` guess, falling back to
                ``application/pdf``.

        Returns:
            A :class:`Document` holding the file's bytes.

        Raises:
            StepError: If the file does not exist or cannot be read.
        """
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise StepError(f"Attachment file could not be read: {p} ({exc})") from exc
        if media_type is None:
            guessed, _ = mimetypes.guess_type(p.name)
            media_type = guessed or DEFAULT_MEDIA_TYPE
        return cls(data=data, media_type=media_type, title=title if title is not None else p.name)

    @property
    def sha256(self) -> str:
        """Hex SHA-256 digest of the document bytes — the cache identity."""
        return hashlib.sha256(self.data).hexdigest()

    def __repr__(self) -> str:  # pragma: no cover - trivial, but keeps bytes out of logs
        return (
            f"Document(title={self.title!r}, media_type={self.media_type!r}, "
            f"bytes={len(self.data)})"
        )


def normalize_attachments(value: Any) -> list[Document]:
    """Coerce an attachments cell into ``list[Document]``.

    Accepts ``None``, a NaN-like blank, a single ``Document`` / path / ``bytes``,
    or a list/tuple of those.

    Args:
        value: Whatever the row held in the attachments column.

    Returns:
        A (possibly empty) list of :class:`Document`.

    Raises:
        StepError: On an unsupported cell type or an unreadable file.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes, os.PathLike, Document)):
        items: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif _is_blank(value):
        return []
    else:
        raise StepError(
            f"Unsupported attachments value of type {type(value).__name__!r}. "
            f"Expected a Document, a file path, bytes, or a list of those."
        )

    docs: list[Document] = []
    for item in items:
        if isinstance(item, Document):
            docs.append(item)
        elif isinstance(item, bytes):
            docs.append(Document(data=item))
        elif isinstance(item, (str, os.PathLike)):
            if isinstance(item, str) and not item.strip():
                continue
            docs.append(Document.from_path(item))
        else:
            raise StepError(
                f"Unsupported attachment entry of type {type(item).__name__!r}. "
                f"Expected a Document, a file path, or bytes."
            )
    return docs


def _is_blank(value: Any) -> bool:
    """True for a missing-value sentinel (``float('nan')``, ``np.nan``, ``pd.NA``).

    Only NaN is unequal to itself.  ``pd.NA`` propagates instead of answering,
    so a comparison result that cannot be coerced to a bool is itself NA.
    Avoids importing pandas — steps stay pandas-free by design.
    """
    try:
        result = value != value  # noqa: PLR0124
    except (TypeError, ValueError):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return True
