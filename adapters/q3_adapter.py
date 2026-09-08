"""Question 3 Integration Adapter.

Provides a clean, stable interface to Question 3's spelling corrector components:
- Vocabulary & Unigram Frequency Model
- Bigram Language Model for contextual probabilities
- Candidate Generation Method A (Standard Edit Distance 1)
- Candidate Generation Method B (Symmetric Delete Spelling Correction)
- Non-word error correction
- Real-word error detection in local context

===============================================================================
IMPORTANT ARCHITECTURAL NOTICE:
At current development time, the actual Question 3 source code and trained models
are not yet supplied by the Q3 team. This module therefore incorporates a
clearly isolated development FALLBACK MODE to enable complete end-to-end
pipeline testing, Speed Demon benchmarking, and Streamlit deployment.

When Question 3 code/models are placed in the project, this adapter automatically
detects and switches to the real Q3 components without modifying any Q4 pipeline code.
===============================================================================
"""

from collections import Counter, defaultdict
import importlib
import logging
import math
import string
import time
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class Q3IntegrationAdapter:
    """Adapter bridging Question 4 to Question 3 spelling correction models."""

    def __init__(
        self,
        model_path: Optional[str] = "models/q3_spelling_model.pkl",
        significance_ratio: float = 8.0,
    ):
        self.model_path = model_path
        self.significance_ratio = significance_ratio
        self.letters = string.ascii_lowercase

        self.using_fallback: bool = True
        self.real_q3_module = None

        # Fallback structures
        self.vocabulary: Set[str] = set()
        self.word_counts: Counter = Counter()
        self.total_words: int = 0
        self.bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.unigram_counts: Dict[str, int] = defaultdict(int)

        # Method B Precomputed Deletion Dictionary: del_variant -> set(original_words)
        self.deletion_dict: Dict[str, Set[str]] = defaultdict(set)

        self._initialize()

    def _initialize(self) -> None:
        """Attempt to load real Q3 implementation, else initialize isolated fallback."""
        # 1. Attempt dynamic import of external Q3 module
        try:
            mod = importlib.import_module("q3_spelling")
            if hasattr(mod, "correct_nonword") and hasattr(mod, "generate_candidates_method_a"):
                self.real_q3_module = mod
                self.using_fallback = False
                logger.info("Successfully hooked into real Question 3 module 'q3_spelling'.")
                return
        except ImportError:
            pass

        # 2. Activate Fallback Mode
        self.using_fallback = True
        logger.warning(
            "\n" + "=" * 78 + "\n"
            "NOTICE: Real Question 3 models are not found on disk/path.\n"
            "ACTIVATING TEMPORARY DEVELOPMENT FALLBACK MODE FOR QUESTION 3.\n"
            "This fallback exists ONLY for Q4 development and demonstration.\n"
            "The final submission MUST replace this with the real Q3 trained models.\n"
            + "=" * 78
        )
        self._init_fallback_models()

    def _init_fallback_models(self) -> None:
        """Build vocabulary, unigrams, bigrams, and Symmetric Delete dictionary from Brown."""
        from nltk.corpus import brown

        sents = brown.sents()
        for sent in sents:
            cleaned = [w.lower() for w in sent if w.isalpha()]
            for i, w in enumerate(cleaned):
                self.word_counts[w] += 1
                self.unigram_counts[w] += 1
                if i > 0:
                    self.bigram_counts[(cleaned[i - 1], w)] += 1

        self.vocabulary = set(self.word_counts.keys())
        self.total_words = sum(self.word_counts.values())

        # Precompute Method B Deletion Dictionary for all words in vocabulary
        for word in self.vocabulary:
            # Generate all 1-character deletions
            for i in range(len(word)):
                del_variant = word[:i] + word[i + 1 :]
                self.deletion_dict[del_variant].add(word)

    def is_known_word(self, word: str) -> bool:
        """Check if word exists in vocabulary."""
        return word.lower() in self.vocabulary

    # -------------------------------------------------------------------------
    # Part 2: Candidate Generation Methods
    # -------------------------------------------------------------------------

    def generate_candidates_method_a(self, word: str) -> Set[str]:
        """Method A: Standard Edit Distance 1 Generation.

        Generates all edits (deletions, transpositions, replacements, insertions)
        and filters for valid vocabulary words.
        """
        w = word.lower()
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]

        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.letters]
        inserts = [L + c + R for L, R in splits for c in self.letters]

        all_edits = set(deletes + transposes + replaces + inserts)
        # Filter for known vocabulary words
        return {cand for cand in all_edits if cand in self.vocabulary and cand != w}

    def generate_candidates_method_b(self, word: str) -> Set[str]:
        """Method B: Symmetric Delete Spelling Correction.

        Highly efficient candidate retrieval using precomputed deletion dictionary.
        """
        w = word.lower()
        candidates: Set[str] = set()

        # 1. Check if word itself was a deletion variant of a longer word (insertion edit)
        if w in self.deletion_dict:
            candidates.update(self.deletion_dict[w])

        # 2. Generate all 1-character deletions of the input word
        for i in range(len(w)):
            del_variant = w[:i] + w[i + 1 :]

            # Check if this deletion variant is an actual vocabulary word (deletion edit)
            if del_variant in self.vocabulary:
                candidates.add(del_variant)

            # Check if this deletion variant matches precomputed deletion variants
            # of vocabulary words (replacement / transposition edits)
            if del_variant in self.deletion_dict:
                for original in self.deletion_dict[del_variant]:
                    # Verify edit distance 1 to avoid spurious matches
                    if abs(len(original) - len(w)) <= 1 and original != w:
                        candidates.add(original)

        candidates.discard(w)
        return candidates

    # -------------------------------------------------------------------------
    # Part 3: Spelling Correction Logic
    # -------------------------------------------------------------------------

    def correct_nonword(
        self, token: str, method: str = "B"
    ) -> Tuple[str, str, float]:
        """Correct a non-word error using highest unigram frequency.

        Args:
            token: Misspelled word.
            method: Candidate generation method ('A' or 'B').

        Returns:
            Tuple of (corrected_token, method_name, latency_ms).
        """
        start = time.perf_counter()
        clean_tok = token.lower().strip(",.?!;:\'\"")

        # If already known, no correction needed
        if clean_tok in self.vocabulary:
            latency = (time.perf_counter() - start) * 1000.0
            return token, "none", latency

        # Generate candidates using requested method
        if method.upper() == "A":
            candidates = self.generate_candidates_method_a(clean_tok)
            method_used = "Method A (Edit-1)"
        else:
            candidates = self.generate_candidates_method_b(clean_tok)
            method_used = "Method B (SymDelete)"

        latency = (time.perf_counter() - start) * 1000.0

        if not candidates:
            return token, "none", latency

        # Select candidate with highest unigram count / probability
        best_cand = max(candidates, key=lambda c: self.word_counts.get(c, 0))

        # Preserve original capitalization if applicable
        if token.istitle():
            best_cand = best_cand.capitalize()
        elif token.isupper():
            best_cand = best_cand.upper()

        return best_cand, method_used, latency

    def detect_real_word_error(
        self, context_tokens: List[str], target_idx: int
    ) -> Optional[Tuple[str, float, float]]:
        """Check if an in-vocabulary word is wrong in context (e.g. 'sea' vs 'see').

        Compares bigram probability of original phrase against edit-1 candidates.
        """
        if not (0 <= target_idx < len(context_tokens)):
            return None

        target_word = context_tokens[target_idx].lower().strip(",.?!;:\'\"")
        if not target_word.isalpha() or len(target_word) < 2:
            return None

        # Generate nearby candidates
        candidates = self.generate_candidates_method_b(target_word)
        if not candidates:
            return None

        # Context words
        prev_word = context_tokens[target_idx - 1].lower() if target_idx > 0 else "<s>"
        next_word = (
            context_tokens[target_idx + 1].lower()
            if target_idx < len(context_tokens) - 1
            else "</s>"
        )

        # Bigram probability of original: P(target | prev) * P(next | target)
        orig_p1 = (self.bigram_counts.get((prev_word, target_word), 0) + 0.05) / (
            self.unigram_counts.get(prev_word, 0) + 0.05 * len(self.vocabulary)
        )
        orig_p2 = (self.bigram_counts.get((target_word, next_word), 0) + 0.05) / (
            self.unigram_counts.get(target_word, 0) + 0.05 * len(self.vocabulary)
        )
        orig_prob = orig_p1 * orig_p2

        best_candidate = None
        best_cand_prob = orig_prob

        for cand in candidates:
            cand_p1 = (self.bigram_counts.get((prev_word, cand), 0) + 0.05) / (
                self.unigram_counts.get(prev_word, 0) + 0.05 * len(self.vocabulary)
            )
            cand_p2 = (self.bigram_counts.get((cand, next_word), 0) + 0.05) / (
                self.unigram_counts.get(cand, 0) + 0.05 * len(self.vocabulary)
            )
            cand_prob = cand_p1 * cand_p2

            if cand_prob > best_cand_prob:
                best_cand_prob = cand_prob
                best_candidate = cand

        # Check significance ratio
        if best_candidate and (best_cand_prob / max(1e-12, orig_prob)) >= self.significance_ratio:
            return best_candidate, orig_prob, best_cand_prob

        return None

    def get_status(self) -> Dict[str, any]:
        """Return diagnostic status of the adapter for the UI."""
        return {
            "is_fallback": self.using_fallback,
            "status_label": "FALLBACK (Dev Mode)" if self.using_fallback else "REAL (Q3 Imported)",
            "vocab_size": len(self.vocabulary),
            "deletion_dict_keys": len(self.deletion_dict),
            "significance_ratio": self.significance_ratio,
        }


# Global adapter instance
default_q3_adapter = Q3IntegrationAdapter()
