"""Tagset Reconciliation Layer.

Reconciles Brown Corpus POS tags (output by Question 1's feature-based classifier)
with Penn Treebank (PTB) POS tags (expected by Question 4's PCFG parser).
Includes logging and statistics collection to measure reconciliation coverage
and quantify any accuracy loss for the final report.
"""

from typing import Dict, List, Tuple
from data.tag_mappings import BROWN_TO_PTB_MAP, PREFIX_FALLBACK_MAP


class TagsetReconciler:
    """Bridges Q1 Brown tags and Q4 Penn Treebank PCFG lexical tags."""

    def __init__(self):
        self.stats = {
            "total_tags_processed": 0,
            "direct_lookup_hits": 0,
            "prefix_fallback_hits": 0,
            "default_fallback_hits": 0,
        }

    def normalize_brown_tag(self, tag: str) -> str:
        """Strip Brown-specific modifiers like -tl, -hl, -nc, negation '*' and compound '+'."""
        if not tag:
            return "nn"

        t = tag.lower().strip()
        # Handle compound tags like 'in+at' or 'wdt+ber' -> take the first component
        if "+" in t:
            t = t.split("+")[0]

        # Strip title (-tl), headline (-hl), foreign (-fw), etc.
        for modifier in ["-tl", "-hl", "-nc", "-fw"]:
            if modifier in t:
                t = t.replace(modifier, "")

        # Strip negation suffix or special marker
        t = t.rstrip("*").strip()
        return t if t else "nn"

    def reconcile_tag(self, brown_tag: str) -> str:
        """Map a single Brown Corpus POS tag to its Penn Treebank equivalent."""
        self.stats["total_tags_processed"] += 1

        if not brown_tag:
            self.stats["default_fallback_hits"] += 1
            return "NN"

        # Direct check for punctuation
        if brown_tag in {".", ",", ":", ";", "--", "-", "(", ")", "[", "]", "?", "!"}:
            self.stats["direct_lookup_hits"] += 1
            return BROWN_TO_PTB_MAP.get(brown_tag, ".")

        norm_tag = self.normalize_brown_tag(brown_tag)

        # 1. Direct dictionary match
        if norm_tag in BROWN_TO_PTB_MAP:
            self.stats["direct_lookup_hits"] += 1
            return BROWN_TO_PTB_MAP[norm_tag]

        # 2. Prefix-based matching
        for prefix, ptb_target in PREFIX_FALLBACK_MAP:
            if norm_tag.startswith(prefix):
                self.stats["prefix_fallback_hits"] += 1
                return ptb_target

        # 3. Default fallback
        self.stats["default_fallback_hits"] += 1
        return "NN"

    def reconcile_tagged_sentence(
        self, tagged_sentence: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Reconcile all tags in a tagged sentence from Brown format to PTB format."""
        return [(word, self.reconcile_tag(tag)) for word, tag in tagged_sentence]

    def get_coverage_stats(self) -> Dict[str, float]:
        """Return percentage breakdown of mapping resolution."""
        total = max(1, self.stats["total_tags_processed"])
        return {
            "total_processed": self.stats["total_tags_processed"],
            "direct_match_pct": (self.stats["direct_lookup_hits"] / total) * 100.0,
            "prefix_match_pct": (self.stats["prefix_fallback_hits"] / total) * 100.0,
            "default_fallback_pct": (self.stats["default_fallback_hits"] / total) * 100.0,
        }

    def reset_stats(self) -> None:
        """Reset the coverage counters."""
        for k in self.stats:
            self.stats[k] = 0


# Global reconciler instance
default_reconciler = TagsetReconciler()
