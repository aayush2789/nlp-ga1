"""Question 1 Integration Adapter.

Provides a clean, stable interface to Question 1's English word segmentation
and feature-based POS tagging components:
- Lexicon / Vocabulary
- Trigram Language Model
- Feature-based POS Classifier
- Joint Beam-Search Decoder

===============================================================================
IMPORTANT ARCHITECTURAL NOTICE:
At current development time, the actual Question 1 source code and trained models
are not yet supplied by the Q1 team. This module therefore incorporates a
clearly isolated development FALLBACK MODE to enable complete end-to-end
pipeline testing and Streamlit deployment.

When Question 1 code/models are placed in the project, this adapter automatically
detects and switches to the real Q1 components without modifying any Q4 pipeline code.
===============================================================================
"""

from collections import Counter, defaultdict
import importlib
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class Q1IntegrationAdapter:
    """Adapter bridging Question 4 to Question 1 segmentation and tagging models."""

    def __init__(
        self,
        model_path: Optional[str] = "models/q1_english_model.pkl",
        alpha: float = 0.7,
        beta: float = 0.3,
        beam_width: int = 10,
        max_word_len: int = 18,
    ):
        self.model_path = model_path
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        self.max_word_len = max_word_len

        self.using_fallback: bool = True
        self.real_q1_module = None

        # Fallback structures
        self.vocabulary: Set[str] = set()
        self.word_freqs: Counter = Counter()
        self.word_tag_map: Dict[str, str] = {}
        self.unigram_log_probs: Dict[str, float] = {}

        self._initialize()

    def _initialize(self) -> None:
        """Attempt to load real Q1 implementation, else initialize isolated fallback."""
        # 1. Attempt dynamic import of external Q1 module
        try:
            mod = importlib.import_module("q1_english")
            if hasattr(mod, "segment_token") and hasattr(mod, "is_known_word"):
                self.real_q1_module = mod
                self.using_fallback = False
                logger.info("Successfully hooked into real Question 1 module 'q1_english'.")
                return
        except ImportError:
            pass

        # 2. Attempt loading pickled model weights if available
        if self.model_path and Path(self.model_path).exists():
            try:
                import pickle
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "decoder" in data:
                    self.real_q1_module = data
                    self.using_fallback = False
                    logger.info(f"Loaded real Question 1 models from '{self.model_path}'.")
                    return
            except Exception as e:
                logger.warning(f"Failed to load Q1 pickle at '{self.model_path}': {e}")

        # 3. Activate Fallback Mode
        self.using_fallback = True
        logger.warning(
            "\n" + "=" * 78 + "\n"
            "NOTICE: Real Question 1 models are not found on disk/path.\n"
            "ACTIVATING TEMPORARY DEVELOPMENT FALLBACK MODE FOR QUESTION 1.\n"
            "This fallback exists ONLY for Q4 development and demonstration.\n"
            "The final submission MUST replace this with the real Q1 trained models.\n"
            + "=" * 78
        )
        self._init_fallback_models()

    def _init_fallback_models(self) -> None:
        """Construct fallback vocabulary, POS tag lookup, and DP decoder from Brown corpus."""
        from nltk.corpus import brown

        tagged_words = brown.tagged_words()
        word_tag_counts = defaultdict(Counter)

        for w, tag in tagged_words:
            w_lower = w.lower()
            self.vocabulary.add(w_lower)
            self.word_freqs[w_lower] += 1
            word_tag_counts[w_lower][tag] += 1

        # Populate most frequent POS tag for each word
        for w, tag_counts in word_tag_counts.items():
            self.word_tag_map[w] = tag_counts.most_common(1)[0][0]

        total_count = sum(self.word_freqs.values())
        for w, count in self.word_freqs.items():
            self.unigram_log_probs[w] = math.log2(count / total_count)

    def is_known_word(self, token: str) -> bool:
        """Check if a token exists in the English vocabulary/lexicon."""
        if not self.using_fallback and hasattr(self.real_q1_module, "is_known_word"):
            return self.real_q1_module.is_known_word(token)

        cleaned = token.lower().strip(",.?!;:\'\"")
        return cleaned in self.vocabulary

    def get_pos_tag(self, word: str) -> str:
        """Predict POS tag for a single word."""
        if not self.using_fallback and hasattr(self.real_q1_module, "get_pos_tag"):
            return self.real_q1_module.get_pos_tag(word)

        w = word.lower().strip(",.?!;:\'\"")
        if w in self.word_tag_map:
            return self.word_tag_map[w]

        # Suffix-based heuristics for unknown tokens in fallback mode
        if w.endswith("ing"):
            return "vbg"
        elif w.endswith("ed"):
            return "vbd"
        elif w.endswith("ly"):
            return "rb"
        elif w.endswith("tion") or w.endswith("ment") or w.endswith("ness"):
            return "nn"
        elif w.endswith("able") or w.endswith("ible") or w.endswith("al"):
            return "jj"
        elif w.endswith("s"):
            return "nns"
        return "nn"

    def tag_sentence(self, words: List[str]) -> List[Tuple[str, str]]:
        """Predict POS tags for a sequence of words."""
        if not self.using_fallback and hasattr(self.real_q1_module, "tag_sentence"):
            return self.real_q1_module.tag_sentence(words)
        return [(w, self.get_pos_tag(w)) for w in words]

    def segment_token(
        self, token: str
    ) -> Tuple[List[str], List[str], float, float]:
        """Perform joint beam-search segmentation on a merged token.

        Args:
            token: Unspaced string candidate (e.g. 'thequick' or 'inmyopinion').

        Returns:
            Tuple of:
            - split_words: Proposed list of segmented words.
            - pos_tags: Predicted POS tags for each word.
            - split_score: Combined log-probability score of the split.
            - single_word_score: Combined score of treating the token as a single word.
        """
        if not self.using_fallback and hasattr(self.real_q1_module, "segment_token"):
            return self.real_q1_module.segment_token(token)

        # Fallback Dynamic Programming Beam-Search Decoder
        clean_token = token.lower().strip(",.?!;:\'\"")
        n = len(clean_token)

        # Baseline single-word score
        single_word_score = self.unigram_log_probs.get(clean_token, -25.0)

        # Beam search table: chart[i] holds top beams of (log_score, [words])
        chart: Dict[int, List[Tuple[float, List[str]]]] = defaultdict(list)
        chart[0] = [(0.0, [])]

        for i in range(n):
            if i not in chart:
                continue
            curr_beams = sorted(chart[i], key=lambda x: x[0], reverse=True)[: self.beam_width]

            max_k = min(n, i + self.max_word_len)
            for j in range(i + 1, max_k + 1):
                word_cand = clean_token[i:j]
                if word_cand in self.vocabulary:
                    lm_log_p = self.unigram_log_probs.get(word_cand, -20.0)
                    tag = self.get_pos_tag(word_cand)
                    pos_log_p = -1.5 if tag else -5.0

                    cand_score = self.alpha * lm_log_p + self.beta * pos_log_p

                    for prev_score, prev_words in curr_beams:
                        new_score = prev_score + cand_score
                        chart[j].append((new_score, prev_words + [word_cand]))

        # Check end of token
        if n in chart and chart[n]:
            best_beams = sorted(chart[n], key=lambda x: x[0], reverse=True)
            for split_score, split_words in best_beams:
                if len(split_words) >= 2:
                    tags = [self.get_pos_tag(w) for w in split_words]
                    # Score of a split is favorable if all sub-words are valid and
                    # split score is competitive
                    return split_words, tags, split_score, single_word_score

        # No valid split into 2+ words found
        return [clean_token], [self.get_pos_tag(clean_token)], single_word_score, single_word_score

    def segment_and_tag(self, token: str) -> List[Tuple[str, str]]:
        """Convenience function returning list of (word, tag) pairs."""
        split_words, tags, split_score, single_score = self.segment_token(token)
        if split_score > single_score and len(split_words) > 1:
            return list(zip(split_words, tags))
        return [(token, self.get_pos_tag(token))]

    def get_status(self) -> Dict[str, any]:
        """Return diagnostic status of the adapter for the UI."""
        return {
            "is_fallback": self.using_fallback,
            "status_label": "FALLBACK (Dev Mode)" if self.using_fallback else "REAL (Q1 Imported)",
            "vocab_size": len(self.vocabulary),
            "alpha": self.alpha,
            "beta": self.beta,
            "beam_width": self.beam_width,
            "max_word_len": self.max_word_len,
        }


# Global adapter instance
default_q1_adapter = Q1IntegrationAdapter()
