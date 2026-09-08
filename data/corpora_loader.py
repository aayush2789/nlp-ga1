"""Corpora Loader and Text Sampler.

Handles loading, tokenizing, and caching text corpora (Brown, Penn Treebank,
Gutenberg, Reuters) needed for language modeling, PCFG training, and live typing simulation.
"""

import random
from typing import List, Optional, Tuple
import nltk
from nltk.corpus import brown, treebank, gutenberg, reuters


def get_brown_sentences(max_sents: Optional[int] = None) -> List[List[str]]:
    """Retrieve tokenized sentences from the Brown Corpus.

    Args:
        max_sents: Max number of sentences to load (None for all).

    Returns:
        List of sentences, where each sentence is a list of string tokens.
    """
    sents = brown.sents()
    if max_sents:
        return list(sents[:max_sents])
    return list(sents)


def get_brown_tagged_words() -> List[Tuple[str, str]]:
    """Retrieve (word, tag) pairs from the Brown Corpus."""
    return brown.tagged_words()


def get_treebank_parsed_sents(max_trees: Optional[int] = None) -> List[nltk.Tree]:
    """Retrieve parsed syntax trees from the Penn Treebank sample.

    Args:
        max_trees: Max number of trees to load (None for all).

    Returns:
        List of nltk.Tree instances.
    """
    trees = treebank.parsed_sents()
    if max_trees:
        return list(trees[:max_trees])
    return list(trees)


def sample_random_passage(
    source: str = "gutenberg",
    min_sentences: int = 5,
    max_sentences: int = 8,
    seed: Optional[int] = None,
) -> Tuple[str, List[str], str]:
    """Randomly sample a contiguous paragraph of 5-8 sentences from a text corpus.

    Args:
        source: Corpus source: 'gutenberg', 'brown', or 'reuters'.
        min_sentences: Minimum number of sentences.
        max_sentences: Maximum number of sentences.
        seed: Optional integer random seed for reproducibility.

    Returns:
        Tuple of (passage_text, sentence_strings, source_file_id).
    """
    rng = random.Random(seed)

    if source == "gutenberg":
        fileids = gutenberg.fileids()
        chosen_file = rng.choice(fileids)
        all_sents = gutenberg.sents(chosen_file)
    elif source == "reuters":
        fileids = reuters.fileids()
        chosen_file = rng.choice(fileids)
        all_sents = reuters.sents(chosen_file)
    else:  # default to brown
        fileids = brown.fileids()
        chosen_file = rng.choice(fileids)
        all_sents = brown.sents(chosen_file)

    num_sents = rng.randint(min_sentences, max_sentences)
    total_sents = len(all_sents)

    if total_sents <= num_sents:
        selected_sents = all_sents
    else:
        start_idx = rng.randint(0, total_sents - num_sents)
        selected_sents = all_sents[start_idx : start_idx + num_sents]

    # Clean and reconstruct sentences
    sentence_strings = []
    for s in selected_sents:
        sent_str = " ".join(s).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
        # Ensure first character is capitalized and ends with punctuation
        sent_str = sent_str.strip()
        if sent_str:
            sentence_strings.append(sent_str)

    passage_text = " ".join(sentence_strings)
    return passage_text, sentence_strings, chosen_file
