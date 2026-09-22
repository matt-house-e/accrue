"""Tests for Document / normalize_attachments (#153)."""

from __future__ import annotations

import hashlib

import pytest

from accrue.core.exceptions import StepError
from accrue.steps.attachments import Document, normalize_attachments

PDF_BYTES = b"%PDF-1.4 fake"


@pytest.fixture
def pdf_file(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(PDF_BYTES)
    return p


# -- Document ------------------------------------------------------------


class TestDocument:
    def test_from_path_defaults(self, pdf_file):
        doc = Document.from_path(pdf_file)
        assert doc.data == PDF_BYTES
        assert doc.title == "report.pdf"
        assert doc.media_type == "application/pdf"

    def test_from_path_explicit_title_and_media_type(self, pdf_file):
        doc = Document.from_path(pdf_file, title="Q3", media_type="application/x-thing")
        assert doc.title == "Q3"
        assert doc.media_type == "application/x-thing"

    def test_from_path_guesses_media_type(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_bytes(b"hi")
        assert Document.from_path(p).media_type == "text/plain"

    def test_from_path_unknown_extension_falls_back_to_pdf(self, tmp_path):
        p = tmp_path / "blob.zzz"
        p.write_bytes(b"hi")
        assert Document.from_path(p).media_type == "application/pdf"

    def test_from_path_accepts_str(self, pdf_file):
        assert Document.from_path(str(pdf_file)).data == PDF_BYTES

    def test_missing_file_raises_step_error(self, tmp_path):
        missing = tmp_path / "nope.pdf"
        with pytest.raises(StepError) as exc:
            Document.from_path(missing)
        assert "nope.pdf" in str(exc.value)

    def test_sha256_matches_hashlib_and_is_stable(self):
        doc = Document(data=PDF_BYTES)
        assert doc.sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
        assert doc.sha256 == Document(data=PDF_BYTES, title="other").sha256

    def test_sha256_differs_on_different_bytes(self):
        assert Document(data=b"a").sha256 != Document(data=b"b").sha256

    def test_repr_has_no_bytes(self):
        doc = Document(data=PDF_BYTES, title="report.pdf")
        text = repr(doc)
        assert "PDF-1.4" not in text
        assert "report.pdf" in text
        assert "application/pdf" in text
        assert str(len(PDF_BYTES)) in text

    def test_frozen(self):
        doc = Document(data=PDF_BYTES)
        with pytest.raises(Exception):
            doc.title = "new"  # type: ignore[misc]


# -- normalize_attachments ------------------------------------------------


class TestNormalizeAttachments:
    def test_none(self):
        assert normalize_attachments(None) == []

    def test_nan(self):
        assert normalize_attachments(float("nan")) == []

    def test_pandas_na(self):
        import pandas as pd

        assert normalize_attachments(pd.NA) == []

    def test_empty_list(self):
        assert normalize_attachments([]) == []

    def test_empty_string(self):
        assert normalize_attachments("") == []
        assert normalize_attachments("   ") == []

    def test_single_document(self):
        doc = Document(data=PDF_BYTES)
        assert normalize_attachments(doc) == [doc]

    def test_single_path_str(self, pdf_file):
        docs = normalize_attachments(str(pdf_file))
        assert len(docs) == 1
        assert docs[0].data == PDF_BYTES
        assert docs[0].title == "report.pdf"

    def test_single_pathlike(self, pdf_file):
        docs = normalize_attachments(pdf_file)
        assert len(docs) == 1 and docs[0].data == PDF_BYTES

    def test_single_bytes(self):
        docs = normalize_attachments(PDF_BYTES)
        assert docs == [Document(data=PDF_BYTES)]
        assert docs[0].media_type == "application/pdf"
        assert docs[0].title is None

    def test_list_of_paths(self, tmp_path):
        a = tmp_path / "a.pdf"
        a.write_bytes(b"A")
        b = tmp_path / "b.pdf"
        b.write_bytes(b"B")
        docs = normalize_attachments([a, str(b)])
        assert [d.data for d in docs] == [b"A", b"B"]

    def test_tuple_of_mixed(self, pdf_file):
        doc = Document(data=b"inline", title="inline")
        docs = normalize_attachments((pdf_file, b"raw", doc))
        assert len(docs) == 3
        assert docs[0].data == PDF_BYTES
        assert docs[1].data == b"raw"
        assert docs[2] is doc

    def test_unsupported_type_raises(self):
        with pytest.raises(StepError) as exc:
            normalize_attachments(42)
        assert "int" in str(exc.value)

    def test_unsupported_entry_type_raises(self):
        with pytest.raises(StepError) as exc:
            normalize_attachments([{"not": "a doc"}])
        assert "dict" in str(exc.value)

    def test_missing_file_in_list_raises(self, tmp_path):
        with pytest.raises(StepError) as exc:
            normalize_attachments([tmp_path / "gone.pdf"])
        assert "gone.pdf" in str(exc.value)
