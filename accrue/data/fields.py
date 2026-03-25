"""Field management for the Accrue enrichment tool.

Loads field categories and specifications from CSV files, producing dicts
compatible with :class:`~lattice.schemas.field_spec.FieldSpec` and
``LLMStep(fields=...)``.

CSV format (v0.3 — 7-key spec)::

    Category,Field,Prompt,Type,Format,Enum,Examples,Bad_Examples,Default

Required columns: ``Category``, ``Field``, ``Prompt``.
Optional columns: ``Type``, ``Format``, ``Enum``, ``Examples``,
``Bad_Examples``, ``Default``.

**Legacy support**: If ``Instructions`` or ``Guidance`` columns exist they
are concatenated into ``prompt``.  If ``Data_Type`` exists and ``Type``
does not, ``Data_Type`` is treated as ``Type``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.exceptions import FieldValidationError

logger = logging.getLogger(__name__)

# Columns that MUST exist in every CSV
REQUIRED_COLUMNS = ["Category", "Field", "Prompt"]


class FieldManager:
    """Loads field categories from CSV and provides structured access."""

    def __init__(self, fields_categories_path: str) -> None:
        self.fields_categories_path = Path(fields_categories_path)
        self.categories: dict[str, dict[str, dict[str, Any]]] = self._load_field_categories(
            fields_categories_path
        )

    @classmethod
    def from_csv(cls, csv_path: str) -> FieldManager:
        return cls(csv_path)

    # -- loading ---------------------------------------------------------

    def _load_field_categories(self, csv_path: str) -> dict[str, dict[str, dict[str, Any]]]:
        try:
            if not os.path.exists(csv_path):
                raise FieldValidationError(f"Fields categories CSV file not found at: {csv_path}")

            df = pd.read_csv(csv_path)

            if df.empty:
                raise FieldValidationError(f"Fields categories CSV file is empty: {csv_path}")

            # Validate required columns
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise FieldValidationError(
                    f"CSV missing required columns: {', '.join(missing)}\n"
                    f"Required: {', '.join(REQUIRED_COLUMNS)}\n"
                    f"Found: {', '.join(df.columns)}"
                )

            categories: dict[str, dict[str, dict[str, Any]]] = {}

            for idx, row in df.iterrows():
                category = row["Category"]
                field = row["Field"]

                if pd.isna(category) or pd.isna(field):
                    logger.warning("Skipping row %d with empty Category or Field", idx)
                    continue

                spec = self._row_to_spec(row, df.columns.tolist())

                categories.setdefault(str(category), {})[str(field)] = spec

            if not categories:
                raise FieldValidationError(f"No valid categories found in CSV file: {csv_path}")

            logger.info("Loaded %d categories from %s", len(categories), csv_path)
            return categories

        except pd.errors.EmptyDataError:
            raise FieldValidationError(f"CSV file is empty or has no data: {csv_path}")
        except pd.errors.ParserError as e:
            raise FieldValidationError(f"Invalid CSV format in {csv_path}: {e}")
        except FieldValidationError:
            raise
        except Exception as e:
            raise FieldValidationError(f"Failed to load field categories CSV {csv_path}: {e}")

    @staticmethod
    def _row_to_spec(row: Any, columns: list[str]) -> dict[str, Any]:
        """Convert a single CSV row to a field spec dict.

        Handles legacy columns (Instructions, Guidance, Data_Type) and
        new 7-key columns (Type, Format, Enum, Examples, Bad_Examples, Default).
        """
        # -- prompt (required) + legacy concatenation
        prompt = str(row["Prompt"]) if pd.notna(row["Prompt"]) else ""

        # Legacy: concatenate Instructions / Guidance into prompt
        for legacy_col in ("Instructions", "Guidance"):
            if legacy_col in columns and pd.notna(row.get(legacy_col)):
                extra = str(row[legacy_col]).strip()
                if extra:
                    prompt = f"{prompt}\n{extra}" if prompt else extra

        spec: dict[str, Any] = {"prompt": prompt}

        # -- type (legacy Data_Type fallback)
        type_val = None
        if "Type" in columns and pd.notna(row.get("Type")):
            type_val = str(row["Type"]).strip()
        elif "Data_Type" in columns and pd.notna(row.get("Data_Type")):
            type_val = str(row["Data_Type"]).strip()
        if type_val:
            spec["type"] = type_val

        # -- format
        if "Format" in columns and pd.notna(row.get("Format")):
            spec["format"] = str(row["Format"]).strip()

        # -- enum (comma-separated or JSON array)
        if "Enum" in columns and pd.notna(row.get("Enum")):
            spec["enum"] = _parse_list_column(row["Enum"])

        # -- examples (comma-separated or JSON array)
        if "Examples" in columns and pd.notna(row.get("Examples")):
            spec["examples"] = _parse_list_column(row["Examples"])

        # -- bad_examples
        if "Bad_Examples" in columns and pd.notna(row.get("Bad_Examples")):
            spec["bad_examples"] = _parse_list_column(row["Bad_Examples"])

        # -- default
        if "Default" in columns and pd.notna(row.get("Default")):
            spec["default"] = str(row["Default"]).strip()

        return spec

    # -- access ----------------------------------------------------------

    def get_category_fields(self, category: str) -> dict[str, dict[str, Any]]:
        """Get field specs for a category.

        Returns all spec keys (prompt, type, format, enum, examples,
        bad_examples, default) — nothing is stripped.
        """
        if category not in self.categories:
            available = ", ".join(self.categories.keys())
            raise FieldValidationError(
                f"Category '{category}' not found. Available categories: {available}"
            )
        # Return copies so callers can't mutate internal state
        return {field: dict(details) for field, details in self.categories[category].items()}

    def get_categories(self) -> list[str]:
        return list(self.categories.keys())

    def validate_category(self, category: str) -> bool:
        return category in self.categories

    def get_field_count(self, category: str | None = None) -> int:
        if category is not None:
            if category not in self.categories:
                return 0
            return len(self.categories[category])
        return sum(len(fields) for fields in self.categories.values())

    def __str__(self) -> str:
        total = self.get_field_count()
        summary = ", ".join(f"{cat}({len(fields)})" for cat, fields in self.categories.items())
        return f"FieldManager({len(self.categories)} categories, {total} total fields: {summary})"


def load_fields(csv_path: str, category: str | None = None) -> dict[str, dict[str, Any]]:
    """Load field specs from CSV, suitable for ``LLMStep(fields=...)``.

    Args:
        csv_path: Path to the field categories CSV file.
        category: If provided, return only fields in this category.

    Returns:
        Dict mapping field names to their 7-key spec dicts.

    Example::

        from accrue.data import load_fields
        fields = load_fields("fields.csv", category="business_analysis")
        pipeline = Pipeline([LLMStep("analyze", fields=fields)])
    """
    fm = FieldManager(csv_path)
    if category:
        return fm.get_category_fields(category)
    all_fields: dict[str, dict[str, Any]] = {}
    for cat in fm.get_categories():
        all_fields.update(fm.get_category_fields(cat))
    return all_fields


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_list_column(value: Any) -> list[str]:
    """Parse a CSV cell into a list of strings.

    Tries JSON array first (``["a", "b"]``), falls back to comma-separated.
    """
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except json.JSONDecodeError:
            pass
    return [s.strip() for s in text.split(",") if s.strip()]
