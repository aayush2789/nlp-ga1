"""Question 1 Integration Adapter.

Provides a clean, stable wrapper directly around Question 1's actual English
word segmentation and POS tagging implementation:
- Lexicon / Vocabulary from Q1 (Brown corpus 80% train split)
- TrigramSegmenter: Dynamic programming / Viterbi trigram segmentation
- TrigramPOSTagger: Dynamic programming / Viterbi emission + transition POS tagger

===============================================================================
INTEGRATION NOTICE:
All temporary development fallback implementations have been removed.
This adapter interfaces directly with the real Question 1 components.
If the Q1 models or dependencies cannot be initialized, an explicit RuntimeError
is raised to prevent silent degradation.
===============================================================================
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from q1.q1_model import (
    TrigramPOSTagger,
    TrigramSegmenter,
    load_english_q1_models,
)

logger = logging.getLogger(__name__)


class Q1IntegrationAdapter:
    """Production integration adapter wrapping the genuine Question 1 English models."""

    def __init__(
        self,
        model_path: Optional[str] = "models/q1_english_model.pkl",
        force_retrain: bool = False,
    ):
        self.model_path = model_path or "models/q1_english_model.pkl"
        self.segmenter: Optional[TrigramSegmenter] = None
        self.tagger: Optional[TrigramPOSTagger] = None
        self.vocabulary: Set[str] = set()

        self._load_q1_components(force_retrain=force_retrain)

    def _load_q1_components(self, force_retrain: bool = False) -> None:
        """Load the genuine Question 1 English models, failing explicitly if unavailable."""
        try:
            self.segmenter, self.tagger, self.vocabulary = load_english_q1_models(
                cache_path=self.model_path, force_retrain=force_retrain
            )
            logger.info(
                f"Successfully loaded genuine Question 1 English models "
                f"(Vocabulary size: {len(self.vocabulary)}, "
                f"Max word len: {self.segmenter.max_word_length}, "
                f"POS tags: {len(self.tagger.tags)})."
            )
        except Exception as e:
            error_msg = (
                f"CRITICAL INTEGRATION FAILURE: Unable to load Question 1 English models "
                f"from '{self.model_path}'. Ensure that the NLTK 'brown' corpus is downloaded "
                f"and Question 1 files are intact. Detailed error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def is_known_word(self, token: str) -> bool:
        """Check if a token exists in the genuine Question 1 English vocabulary."""
        cleaned = token.lower().strip(",.?!;:\'\"")
        return cleaned in self.vocabulary

    def get_pos_tag(self, word: str) -> str:
        """Predict POS tag for a single word using Question 1's TrigramPOSTagger."""
        cleaned = word.lower().strip(",.?!;:\'\"")
        if not cleaned:
            return "nn"

        tags = self.tagger.tag([cleaned])
        return tags[0] if tags else "nn"

    def tag_sentence(self, words: List[str]) -> List[Tuple[str, str]]:
        """Predict POS tags for an entire word sequence using Question 1's TrigramPOSTagger."""
        cleaned_words = [w.lower().strip(",.?!;:\'\"") for w in words]
        tags = self.tagger.tag(cleaned_words)
        return list(zip(words, tags))

    def segment_token(
        self, token: str
    ) -> Tuple[List[str], List[str], float, float]:
        """Perform word segmentation on an unspaced or merged token using Q1 TrigramSegmenter.

        Calls the actual Q1 english_segmenter.segment() dynamic programming algorithm.

        Args:
            token: Unspaced string candidate (e.g. 'thequick' or 'doctorexamined').

        Returns:
            Tuple of:
            - split_words: Proposed list of segmented words from Q1.
            - pos_tags: Predicted POS tags for each word from Q1 TrigramPOSTagger.
            - split_score: 1.0 if a valid multi-word split was identified, else 0.0.
            - single_word_score: 0.0 baseline score.
        """
        clean_tok = token.lower().strip(",.?!;:\'\"")
        if not clean_tok:
            return [token], [self.get_pos_tag(token)], 0.0, 0.0

        # Call genuine Q1 TrigramSegmenter
        split_words = self.segmenter.segment(clean_tok)

        # A split is valid if Q1 partitioned the token into 2+ words and every word
        # exists in Q1's English vocabulary (distinguishing real splits from unsegmented single chars)
        is_valid_multi_word_split = (
            len(split_words) >= 2
            and all(self.is_known_word(w) for w in split_words)
        )

        if is_valid_multi_word_split:
            tags = self.tagger.tag(split_words)
            return split_words, tags, 1.0, 0.0

        # Retained as a single token
        tag = self.get_pos_tag(clean_tok)
        return [clean_tok], [tag], 0.0, 0.0

    def segment_and_tag(self, token: str) -> List[Tuple[str, str]]:
        """Perform Q1 segmentation and Q1 POS tagging, returning (word, tag) pairs."""
        split_words, tags, split_score, single_score = self.segment_token(token)
        return list(zip(split_words, tags))

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status of the genuine Q1 integration adapter for UI and telemetry."""
        return {
            "is_fallback": False,
            "status_label": "REAL (Q1 Imported)",
            "vocab_size": len(self.vocabulary),
            "max_word_len": self.segmenter.max_word_length if self.segmenter else 0,
            "num_tags": len(self.tagger.tags) if self.tagger else 0,
            "training_tokens": self.segmenter.total_words if self.segmenter else 0,
        }


# Global adapter instance initialized with genuine Question 1 models
default_q1_adapter = Q1IntegrationAdapter()
