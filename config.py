"""Central Configuration Module for Question 4 Integrated Background Editor.

Defines all tunable hyperparameters, threshold settings, and path configurations
for live word segmentation, spelling correction, PCFG parsing, and smoothed n-gram
language modeling.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EditorConfig:
    """Configuration settings for the Question 4 Integrated Editor."""

    # -------------------------------------------------------------------------
    # 1. Fast-Typing Simulation Settings
    # -------------------------------------------------------------------------
    # Probability 'p' of dropping a space between two consecutive words during
    # fast-typing simulation (mimics a typist occasionally failing to hit spacebar).
    # Value justified: p=0.08 produces ~1-2 merges per 15-20 words, giving Q1
    # segmentation realistic work without completely degrading sentence structure.
    merge_probability_p: float = 0.08

    # Simulated delay (in seconds) between streaming incoming tokens in live demo
    typing_delay_sec: float = 0.08

    # Random seed for reproducible demonstrations (None for random passage)
    random_seed: Optional[int] = None

    # Passage size in sentences when sampling from Gutenberg / Brown / Reuters
    min_sentences: int = 5
    max_sentences: int = 8

    # -------------------------------------------------------------------------
    # 2. Pipeline Trigger & Latency Intervals
    # -------------------------------------------------------------------------
    # Word-count trigger interval 'N' for periodic grammar and real-word error checks.
    # Value justified: N=5 provides regular real-time feedback after every short phrase
    # while keeping computational overhead well below the typing rate.
    grammar_trigger_interval_N: int = 5

    # Lookback window (in words) evaluated during each periodic grammar check
    grammar_window_size: int = 8

    # Perplexity threshold for flagging a local phrase as implausible [GRAMMAR-ALERT]
    grammar_perplexity_threshold: float = 850.0

    # -------------------------------------------------------------------------
    # 3. Question 1: Word Segmentation & POS Tagging
    # -------------------------------------------------------------------------
    # Maximum word length allowed during segmentation splitting
    max_word_length_segmentation: int = 18

    # Suspicious word length threshold: tokens longer than this trigger segmentation
    # check even if they match a rare long token
    suspicious_word_length: int = 14

    # Beam width for joint segmentation decoder
    segmentation_beam_width: int = 10

    # Weight parameters inherited from Q1 formulation:
    # Score = alpha * log P_LM(words) + beta * log P_POS(tags | words)
    segmentation_alpha: float = 0.7
    segmentation_beta: float = 0.3

    # External Q1 model path or module name for dynamic loading
    q1_model_path: Optional[str] = "models/q1_english_model.pkl"

    # -------------------------------------------------------------------------
    # 4. Question 3: Spelling Correction
    # -------------------------------------------------------------------------
    # Preferred candidate generation method: 'B' (Symmetric Delete) or 'A' (Edit-1)
    # Value justified: Method B achieves ~10x-50x speedup via precomputed deletion dict
    spelling_candidate_method: str = "B"

    # Max edit distance for spelling correction
    spelling_max_edit_distance: int = 1

    # Real-word error likelihood ratio: if P(candidate_phrase) / P(original_phrase) > ratio,
    # then flag as a real-word error under [GRAMMAR-ALERT]
    real_word_error_significance_ratio: float = 8.0

    # External Q3 model path or module name for dynamic loading
    q3_model_path: Optional[str] = "models/q3_spelling_model.pkl"

    # -------------------------------------------------------------------------
    # 5. Question 4: Shared Smoothed Language Models
    # -------------------------------------------------------------------------
    # Add-k smoothing parameter 'k'.
    # Value justified: k=0.05 provides optimal balance on Brown corpus, preventing
    # zero-frequency penalties for unseen transitions without flattening probabilities.
    smoothing_k: float = 0.05

    # Number of Brown corpus sentences used for training language models
    # (Use 25,000 sentences for balanced training & fast load time)
    brown_train_sentences: int = 25000

    # -------------------------------------------------------------------------
    # 6. Question 4: PCFG Constituency Parser
    # -------------------------------------------------------------------------
    # Number of Penn Treebank trees used for PCFG induction (sample size)
    treebank_sample_size: int = 1500

    # Max sentence length (tokens) for PCFG CKY parsing.
    # Sentences longer than this are gracefully delegated to the Trigram/Bigram LM
    # to avoid O(n^3) CKY latency spikes during live analysis.
    pcfg_max_sentence_length: int = 22

    # Timeout in seconds for individual sentence PCFG parsing
    pcfg_parse_timeout_sec: float = 1.5

    # PCFG log probability threshold below which a sentence is deemed questionable
    pcfg_log_prob_threshold: float = -65.0

    # -------------------------------------------------------------------------
    # 7. Speed Demon Benchmark Settings
    # -------------------------------------------------------------------------
    speed_demon_batch_size: int = 1000

    # Base workspace directory
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)


# Default global instance
default_config = EditorConfig()
