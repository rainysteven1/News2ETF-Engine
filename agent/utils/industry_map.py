"""Industry mapping utilities — loads industry_dict.json and provides sub_category -> industry mapping."""

from __future__ import annotations

import json
from pathlib import Path


class IndustryMapper:
    """Maps sub_category to major industry based on industry_dict.json."""

    def __init__(self, dict_path: Path):
        with open(dict_path, encoding="utf-8") as f:
            self._raw: dict = json.load(f)

        # Build reverse lookup: sub_category -> industry (major category)
        self._sub_to_industry: dict[str, str] = {}
        self._industry_to_subs: dict[str, list[str]] = {}
        self._all_industries: list[str] = list(self._raw.keys())

        for industry, categories in self._raw.items():
            self._industry_to_subs[industry] = []
            for sub_category, indexes in categories.items():
                for idx in indexes:
                    self._sub_to_industry[idx] = industry
                self._industry_to_subs[industry].append(sub_category)

    def get_industry(self, sub_category: str) -> str | None:
        """Return the major industry for a sub_category, or None if not found."""
        return self._sub_to_industry.get(sub_category)

    def get_sub_categories(self, industry: str) -> list[str]:
        """Return all sub_categories for a given major industry."""
        return self._industry_to_subs.get(industry, [])

    @property
    def industries(self) -> list[str]:
        """Return list of all major industries."""
        return self._all_industries

    def industry_etfs(self, industry: str) -> list[str]:
        """Return all ETF/index names for a given industry."""
        categories = self._raw.get(industry, {})
        etfs = []
        for subs in categories.values():
            etfs.extend(subs)
        return etfs
