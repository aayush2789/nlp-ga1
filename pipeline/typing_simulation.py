"""Typing Simulation and Fast-Typing Merge Generator.

Simulates a human typist streaming text word-by-word with small delays,
injecting fast-typing merge errors with probability 'p' (dropping spaces between
consecutive words) to test Question 1's segmentation decoder.
"""

from dataclasses import dataclass
import random
import re
import time
from typing import Generator, List, Optional, Tuple
from data.corpora_loader import sample_random_passage


@dataclass
class InjectedMerge:
    """Tracks an artificially injected word merge for ground truth evaluation."""
    token_idx: int
    merged_token: str
    original_words: List[str]
    sentence_idx: int


class TypingSimulator:
    """Manages passage selection, fast-typing merge corruption, and streaming."""

    def __init__(
        self,
        p: float = 0.08,
        delay_sec: float = 0.08,
        seed: Optional[int] = None,
    ):
        self.p = p
        self.delay_sec = delay_sec
        self.seed = seed
        self.rng = random.Random(seed)

    def prepare_passage(
        self,
        passage_text: Optional[str] = None,
        source: str = "gutenberg",
        min_sentences: int = 5,
        max_sentences: int = 8,
    ) -> Tuple[str, List[str], List[str], List[InjectedMerge]]:
        """Prepare a passage by splitting sentences and introducing fast-typing merges.

        Args:
            passage_text: Custom text (if provided) or None to sample automatically.
            source: Corpus source ('gutenberg', 'brown', or 'reuters') if sampling.
            min_sentences: Min sentences if sampling.
            max_sentences: Max sentences if sampling.

        Returns:
            Tuple of:
            - original_passage: Raw unaltered text.
            - original_sentences: List of original sentence strings.
            - corrupted_tokens: Token stream containing merged words.
            - injected_merges: List of InjectedMerge tracking instances.
        """
        if passage_text:
            # Simple sentence splitting on user-provided text
            raw_sents = re.split(r"(?<=[.!?])\s+", passage_text.strip())
            original_sentences = [s.strip() for s in raw_sents if s.strip()]
            original_passage = passage_text
        else:
            original_passage, original_sentences, _ = sample_random_passage(
                source=source,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                seed=self.seed,
            )

        corrupted_tokens: List[str] = []
        injected_merges: List[InjectedMerge] = []

        global_token_idx = 0
        for sent_idx, sent in enumerate(original_sentences):
            words = sent.split()
            i = 0
            while i < len(words):
                # Check if we should merge with next word (if not punctuation and not last word)
                if (
                    i < len(words) - 1
                    and words[i].isalpha()
                    and words[i + 1].isalpha()
                    and self.rng.random() < self.p
                ):
                    merged_word = words[i] + words[i + 1]
                    corrupted_tokens.append(merged_word)
                    injected_merges.append(
                        InjectedMerge(
                            token_idx=global_token_idx,
                            merged_token=merged_word,
                            original_words=[words[i], words[i + 1]],
                            sentence_idx=sent_idx,
                        )
                    )
                    global_token_idx += 1
                    i += 2  # Consumed both words
                else:
                    corrupted_tokens.append(words[i])
                    global_token_idx += 1
                    i += 1

        return original_passage, original_sentences, corrupted_tokens, injected_merges

    def stream_tokens(
        self, tokens: List[str], delay: Optional[float] = None
    ) -> Generator[Tuple[int, str], None, None]:
        """Yield tokens one by one with optional time.sleep to simulate live typing."""
        d = self.delay_sec if delay is None else delay
        for idx, token in enumerate(tokens):
            if d > 0:
                time.sleep(d)
            yield idx, token
