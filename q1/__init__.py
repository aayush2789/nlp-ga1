"""Question 1 Word Segmentation and POS Tagging Package."""

from q1.q1_model import (
    TrigramSegmenter,
    TrigramPOSTagger,
    build_word_model,
    load_english_q1_models,
)

__all__ = [
    "TrigramSegmenter",
    "TrigramPOSTagger",
    "build_word_model",
    "load_english_q1_models",
]
