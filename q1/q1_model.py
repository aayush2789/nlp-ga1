"""Question 1 Model Implementation and English Model Loader.

Extracted directly from the Question 1 solution (q1.py).
Contains:
- build_word_model: Builds word, bigram, and trigram frequencies and lexicon.
- TrigramSegmenter: Dynamic programming / Viterbi word segmentation using trigram LM.
- TrigramPOSTagger: Dynamic programming / Viterbi POS tagger using emission and trigram transitions.
- load_english_q1_models: Loads or trains the genuine English Q1 models on Brown 80/20 split (seed 42).
"""

from collections import Counter, defaultdict
import math
from pathlib import Path
import pickle
import random
from typing import Dict, List, Optional, Set, Tuple
import nltk


def build_word_model(sentences: List[List[Tuple[str, str]]]) -> Tuple[Counter, Counter, Counter, Set[str]]:
    """Build word, bigram, and trigram count models from tagged sentences."""
    word_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    vocabulary = set()

    for sentence in sentences:
        words = []
        for item in sentence:
            if isinstance(item, tuple):
                word = item[0]
            elif isinstance(item, dict):
                word = item["word"]
            else:
                word = str(item)

            word = word.lower()
            if word.isalpha():
                words.append(word)
                word_counts[word] += 1
                vocabulary.add(word)

        padded = ["<START>", "<START>"] + words + ["<END>"]

        # Bigram counts
        for i in range(1, len(padded)):
            bigram_counts[(padded[i - 1], padded[i])] += 1

        # Trigram counts
        for i in range(2, len(padded)):
            trigram_counts[(padded[i - 2], padded[i - 1], padded[i])] += 1

    return word_counts, bigram_counts, trigram_counts, vocabulary


class TrigramSegmenter:
    """Question 1 Dynamic Programming / Viterbi Trigram Word Segmenter."""

    def __init__(
        self,
        word_counts: Counter,
        bigram_counts: Counter,
        trigram_counts: Counter,
        vocabulary: Set[str],
    ):
        self.word_counts = word_counts
        self.bigram_counts = bigram_counts
        self.trigram_counts = trigram_counts
        self.vocabulary = vocabulary
        self.total_words = sum(word_counts.values())

        if vocabulary:
            self.max_word_length = max(len(word) for word in vocabulary)
        else:
            self.max_word_length = 20

    def word_probability(self, word: str) -> float:
        numerator = self.word_counts[word] + 1
        denominator = self.total_words + len(self.vocabulary)
        return numerator / denominator

    def trigram_probability(self, previous_previous: str, previous: str, current: str) -> float:
        numerator = self.trigram_counts[(previous_previous, previous, current)] + 1
        denominator = self.bigram_counts[(previous_previous, previous)] + len(self.vocabulary) + 1
        return numerator / denominator

    def get_candidates(self, text: str, position: int) -> List[str]:
        candidates = []
        max_length = min(self.max_word_length, len(text) - position)

        for end in range(position + 1, position + max_length + 1):
            word = text[position:end].lower()
            if word in self.vocabulary:
                candidates.append(word)

        return candidates

    def segment(self, text: str) -> List[str]:
        text = text.lower()
        n = len(text)
        if n == 0:
            return []

        dp = {(0, "<START>", "<START>"): (0.0, [])}

        for position in range(n + 1):
            current_states = [state for state in dp if state[0] == position]

            for state in current_states:
                pos, prev_prev, prev = state
                score, words_so_far = dp[state]

                if pos == n:
                    continue

                candidates = self.get_candidates(text, pos)
                if not candidates:
                    candidates = [text[pos : pos + 1]]

                for word in candidates:
                    new_position = pos + len(word)
                    word_prob = self.word_probability(word)
                    transition_prob = self.trigram_probability(prev_prev, prev, word)
                    new_score = score + math.log(word_prob) + math.log(transition_prob)
                    new_state = (new_position, prev, word)

                    if new_state not in dp or new_score > dp[new_state][0]:
                        dp[new_state] = (new_score, words_so_far + [word])

        best_score = float("-inf")
        best_words = None

        for state, value in dp.items():
            position, prev_prev, prev = state
            if position != n:
                continue

            score, words_so_far = value
            end_probability = self.trigram_probability(prev_prev, prev, "<END>")
            final_score = score + math.log(end_probability)

            if final_score > best_score:
                best_score = final_score
                best_words = words_so_far

        return best_words if best_words is not None else [text]


class TrigramPOSTagger:
    """Question 1 Dynamic Programming / Viterbi Trigram POS Tagger."""

    def __init__(self):
        self.word_tag_counts = Counter()
        self.tag_counts = Counter()
        self.bigram_tag_counts = Counter()
        self.trigram_tag_counts = Counter()
        self.tags = set()
        self.vocabulary = set()

    def train(self, sentences: List[List[Tuple[str, str]]]) -> None:
        for sentence in sentences:
            tags = []
            for word, tag in sentence:
                word = word.lower()
                tags.append(tag)
                self.word_tag_counts[(word, tag)] += 1
                self.tag_counts[tag] += 1
                self.tags.add(tag)
                self.vocabulary.add(word)

            padded_tags = ["<START>", "<START>"] + tags + ["<END>"]

            for i in range(1, len(padded_tags)):
                self.bigram_tag_counts[(padded_tags[i - 1], padded_tags[i])] += 1

            for i in range(2, len(padded_tags)):
                self.trigram_tag_counts[(padded_tags[i - 2], padded_tags[i - 1], padded_tags[i])] += 1

    def emission_probability(self, word: str, tag: str) -> float:
        numerator = self.word_tag_counts[(word, tag)] + 1
        denominator = self.tag_counts[tag] + len(self.vocabulary)
        return numerator / denominator

    def transition_probability(self, previous_previous_tag: str, previous_tag: str, current_tag: str) -> float:
        numerator = self.trigram_tag_counts[(previous_previous_tag, previous_tag, current_tag)] + 1
        denominator = self.bigram_tag_counts[(previous_previous_tag, previous_tag)] + len(self.tags) + 1
        return numerator / denominator

    def get_possible_tags(self, word: str) -> Set[str]:
        possible_tags = set()
        for tag in self.tags:
            if self.word_tag_counts[(word, tag)] > 0:
                possible_tags.add(tag)

        if not possible_tags:
            possible_tags = self.tags

        return possible_tags

    def tag(self, words: List[str]) -> List[str]:
        words = [word.lower() for word in words]
        if len(words) == 0:
            return []

        dp = {("<START>", "<START>"): (0.0, [])}

        for word in words:
            new_dp = {}
            possible_tags = self.get_possible_tags(word)

            for (prev_prev, prev), (score, tag_sequence) in dp.items():
                for tag in possible_tags:
                    emission = math.log(self.emission_probability(word, tag))
                    transition = math.log(self.transition_probability(prev_prev, prev, tag))
                    new_score = score + emission + transition
                    state = (prev, tag)

                    if state not in new_dp or new_score > new_dp[state][0]:
                        new_dp[state] = (new_score, tag_sequence + [tag])

            dp = new_dp

        best_score = float("-inf")
        best_tags = None

        for (prev_prev, prev), (score, tag_sequence) in dp.items():
            end_probability = self.transition_probability(prev_prev, prev, "<END>")
            final_score = score + math.log(end_probability)
            if final_score > best_score:
                best_score = final_score
                best_tags = tag_sequence

        return best_tags if best_tags is not None else ["nn"] * len(words)


def load_english_q1_models(
    cache_path: str = "models/q1_english_model.pkl",
    force_retrain: bool = False,
) -> Tuple[TrigramSegmenter, TrigramPOSTagger, Set[str]]:
    """Load or train the actual Question 1 English segmentation and POS tagging models."""
    cache_file = Path(cache_path)

    if not force_retrain and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if (
                isinstance(data, dict)
                and "segmenter" in data
                and "tagger" in data
                and "vocabulary" in data
            ):
                return data["segmenter"], data["tagger"], data["vocabulary"]
        except Exception:
            pass

    # Train actual Question 1 English models using Brown Corpus 80/20 split (seed 42)
    from nltk.corpus import brown

    english_sentences = list(brown.tagged_sents())
    english_sentences = [sentence for sentence in english_sentences if len(sentence) > 0]

    # Exact Q1 reproducibility settings
    random.seed(42)
    random.shuffle(english_sentences)
    english_split = int(0.8 * len(english_sentences))
    english_train = english_sentences[:english_split]

    # Train Q1 segmenter
    (
        english_word_counts,
        english_bigram_counts,
        english_trigram_counts,
        english_vocabulary,
    ) = build_word_model(english_train)

    segmenter = TrigramSegmenter(
        english_word_counts,
        english_bigram_counts,
        english_trigram_counts,
        english_vocabulary,
    )

    # Train Q1 tagger
    tagger = TrigramPOSTagger()
    tagger.train(english_train)

    # Cache for instant subsequent startups
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(
            {
                "segmenter": segmenter,
                "tagger": tagger,
                "vocabulary": english_vocabulary,
            },
            f,
        )

    return segmenter, tagger, english_vocabulary
