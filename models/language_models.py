"""Shared Smoothed N-gram Language Models (Bigram and Trigram).

Implements add-k smoothed Bigram and Trigram models trained on the Brown Corpus.
Reused across segmentation scoring, real-word spelling error detection,
live grammar alerting, and final sentence-level analysis.
"""

from collections import Counter, defaultdict
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple


class SmoothedNgramModel:
    """N-gram language model with add-k smoothing and sentence boundary handling."""

    def __init__(self, n: int = 2, k: float = 0.05, min_freq: int = 1):
        """Initialize the n-gram model.

        Args:
            n: Order of the model (2 for bigram, 3 for trigram).
            k: Add-k smoothing parameter.
            min_freq: Minimum unigram frequency to keep word in vocabulary (else <UNK>).
        """
        self.n = n
        self.k = k
        self.min_freq = min_freq

        self.vocab: Set[str] = set()
        self.unigram_counts: Counter = Counter()
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.total_tokens: int = 0
        self.vocab_size: int = 0
        self.is_trained: bool = False

    def _pad_tokens(self, tokens: List[str]) -> List[str]:
        """Pad token sequence with boundary symbols."""
        pads = ["<s>"] * (self.n - 1)
        return pads + tokens + ["</s>"]

    def _clean_token(self, token: str) -> str:
        """Map word to lowercase and replace rare words with <UNK>."""
        t = token.lower()
        if self.is_trained and t not in self.vocab:
            return "<UNK>"
        return t

    def train(self, sentences: Iterable[List[str]]) -> None:
        """Train the n-gram model on a corpus of sentences."""
        raw_unigrams: Counter = Counter()
        tokenized_sents: List[List[str]] = []

        for sent in sentences:
            if not sent:
                continue
            cleaned = [w.lower() for w in sent]
            raw_unigrams.update(cleaned)
            tokenized_sents.append(cleaned)

        # Build vocabulary with min_freq threshold
        self.vocab = {w for w, c in raw_unigrams.items() if c >= self.min_freq}
        self.vocab.add("<UNK>")
        self.vocab.add("<s>")
        self.vocab.add("</s>")
        self.vocab_size = len(self.vocab)

        # Count n-grams
        for sent in tokenized_sents:
            padded = self._pad_tokens([w if w in self.vocab else "<UNK>" for w in sent])
            self.total_tokens += len(sent)

            for i in range(len(padded)):
                # Count unigrams
                self.unigram_counts[padded[i]] += 1

                # Count n-grams
                if i >= self.n - 1:
                    ngram = tuple(padded[i - self.n + 1 : i + 1])
                    context = ngram[:-1]
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1

        self.is_trained = True

    def prob(self, word: str, context: Tuple[str, ...]) -> float:
        """Calculate conditional probability P(word | context) with add-k smoothing."""
        cleaned_word = self._clean_token(word)
        cleaned_context = tuple(self._clean_token(w) for w in context)

        ngram = cleaned_context + (cleaned_word,)
        count_ngram = self.ngram_counts.get(ngram, 0)
        count_context = self.context_counts.get(cleaned_context, 0)

        # Add-k smoothing formula: P = (C(ngram) + k) / (C(context) + k * |V|)
        prob = (count_ngram + self.k) / (count_context + self.k * self.vocab_size)
        return prob

    def log_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """Return base-2 log probability log2 P(word | context)."""
        p = self.prob(word, context)
        return math.log2(p)

    def sentence_log_prob(self, tokens: List[str]) -> float:
        """Calculate total base-2 log probability of an entire sentence."""
        if not tokens:
            return -float("inf")

        padded = self._pad_tokens([self._clean_token(w) for w in tokens])
        total_log_prob = 0.0

        for i in range(self.n - 1, len(padded)):
            word = padded[i]
            context = tuple(padded[i - self.n + 1 : i])
            total_log_prob += self.log_prob(word, context)

        return total_log_prob

    def sentence_perplexity(self, tokens: List[str]) -> float:
        """Calculate perplexity of a sentence under the model."""
        if not tokens:
            return float("inf")

        log_p = self.sentence_log_prob(tokens)
        # Sequence length includes tokens plus </s> end marker
        m = len(tokens) + 1
        entropy = -log_p / m
        try:
            return math.pow(2, min(entropy, 50.0))  # Cap to prevent float overflow
        except OverflowError:
            return float("inf")

    def local_phrase_log_prob(
        self, tokens: List[str], target_idx: int, replacement: Optional[str] = None
    ) -> float:
        """Calculate log probability of a local window around target_idx.

        Used for real-word error comparison between original word and candidate.
        """
        words = list(tokens)
        if replacement is not None and 0 <= target_idx < len(words):
            words[target_idx] = replacement

        # Compute log probability in window around target_idx
        start = max(0, target_idx - 2)
        end = min(len(words), target_idx + 3)
        window = words[start:end]

        return self.sentence_log_prob(window)


class SharedLanguageModels:
    """Singleton-style wrapper holding shared Bigram and Trigram models."""

    _instance: Optional["SharedLanguageModels"] = None

    def __init__(self, k: float = 0.05, max_sents: int = 25000):
        from data.corpora_loader import get_brown_sentences

        sents = get_brown_sentences(max_sents=max_sents)
        self.bigram_model = SmoothedNgramModel(n=2, k=k)
        self.bigram_model.train(sents)

        self.trigram_model = SmoothedNgramModel(n=3, k=k)
        self.trigram_model.train(sents)

        self.vocabulary: Set[str] = self.bigram_model.vocab

    @classmethod
    def get_instance(cls, k: float = 0.05, max_sents: int = 25000) -> "SharedLanguageModels":
        if cls._instance is None:
            cls._instance = cls(k=k, max_sents=max_sents)
        return cls._instance
