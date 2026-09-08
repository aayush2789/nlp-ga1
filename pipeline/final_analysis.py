"""Final Passage Analysis and Method Comparison.

Splits the corrected passage into sentences and scores each with:
1. PCFG parse log-probability (or 'unparseable')
2. Bigram log-probability & perplexity
3. Trigram log-probability & perplexity

Applies a documented, explainable decision rule to select the most suitable
method per sentence and output a final grammaticality verdict, along with
the count of segmentation merges resolved and spelling corrections applied.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from config import EditorConfig, default_config
from models.language_models import SharedLanguageModels
from models.pcfg_parser import PCFGManager


@dataclass
class SentenceAnalysisRecord:
    """Record of sentence-level evaluation across PCFG and N-gram models."""
    sentence_idx: int
    sentence_text: str
    pcfg_result: str
    pcfg_log_prob: float
    bigram_score: float
    bigram_ppl: float
    trigram_score: float
    trigram_ppl: float
    chosen_method: str
    decision_reason: str
    final_verdict: str  # 'Grammatical', 'Questionable', 'Ungrammatical'
    merges_resolved: int
    spelling_corrections: int
    tree_str: Optional[str] = None


class FinalPassageAnalyzer:
    """Performs end-of-passage sentence extraction, scoring, and verdict synthesis."""

    def __init__(self, config: Optional[EditorConfig] = None):
        self.config = config or default_config
        self.lm = SharedLanguageModels.get_instance(k=self.config.smoothing_k)
        self.pcfg_mgr = PCFGManager.get_instance(sample_size=self.config.treebank_sample_size)

    def split_into_sentences(self, tokens: List[str]) -> List[List[str]]:
        """Group a list of tokens into sentences based on terminating punctuation."""
        sentences: List[List[str]] = []
        current_sent: List[str] = []

        for token in tokens:
            current_sent.append(token)
            # If token ends with sentence-terminating punctuation
            if re.search(r"[.!?]$", token):
                sentences.append(current_sent)
                current_sent = []

        if current_sent:
            sentences.append(current_sent)

        return sentences

    def evaluate_sentence(
        self,
        sent_idx: int,
        tokens: List[str],
        tagged_words: List[Tuple[str, str]],
        merges_count: int = 0,
        spelling_count: int = 0,
    ) -> SentenceAnalysisRecord:
        """Score a single sentence and apply the multi-tier decision rule."""
        sent_str = " ".join(tokens)

        # 1. PCFG Constituency Parse
        parse_res = self.pcfg_mgr.parser.parse(
            tagged_words, max_sentence_len=self.config.pcfg_max_sentence_length
        )

        pcfg_result_str = (
            f"{parse_res.log_prob:.2f}" if parse_res.is_valid else "unparseable"
        )
        pcfg_log_p = parse_res.log_prob if parse_res.is_valid else -float("inf")

        # 2. Bigram LM Scoring
        bigram_log_p = self.lm.bigram_model.sentence_log_prob(tokens)
        bigram_ppl = self.lm.bigram_model.sentence_perplexity(tokens)

        # 3. Trigram LM Scoring
        trigram_log_p = self.lm.trigram_model.sentence_log_prob(tokens)
        trigram_ppl = self.lm.trigram_model.sentence_perplexity(tokens)

        # ---------------------------------------------------------------------
        # Documented Decision Rule:
        # Tier 1: Prefer PCFG when valid parse exists and log prob is not an outlier
        # Tier 2: Prefer Trigram LM if low perplexity and valid transitions
        # Tier 3: Fallback to Bigram LM for local sequence judgment
        # ---------------------------------------------------------------------
        if parse_res.is_valid and pcfg_log_p >= self.config.pcfg_log_prob_threshold:
            chosen_method = "PCFG Parser"
            verdict = "Grammatical"
            reason = (
                f"Valid hierarchical syntax tree found with log-prob {pcfg_log_p:.2f} "
                f"(>= threshold {self.config.pcfg_log_prob_threshold:.1f})."
            )
        elif trigram_ppl <= 350.0:
            chosen_method = "Trigram LM"
            verdict = "Grammatical"
            reason = (
                f"PCFG parse unavailable or low-probability ({pcfg_result_str}); "
                f"Trigram perplexity ({trigram_ppl:.1f}) demonstrates natural phrasing."
            )
        elif bigram_ppl <= 500.0:
            chosen_method = "Bigram LM"
            verdict = "Questionable"
            reason = (
                f"Both PCFG and Trigram indicated high structural irregularity; "
                f"Bigram perplexity ({bigram_ppl:.1f}) reflects plausible local bigrams."
            )
        else:
            chosen_method = "Bigram/Trigram LM"
            verdict = "Ungrammatical"
            reason = (
                f"PCFG unparseable and severe perplexity spikes in both Bigram ({bigram_ppl:.1f}) "
                f"and Trigram ({trigram_ppl:.1f}) models."
            )

        return SentenceAnalysisRecord(
            sentence_idx=sent_idx + 1,
            sentence_text=sent_str,
            pcfg_result=pcfg_result_str,
            pcfg_log_prob=pcfg_log_p,
            bigram_score=bigram_log_p,
            bigram_ppl=bigram_ppl,
            trigram_score=trigram_log_p,
            trigram_ppl=trigram_ppl,
            chosen_method=chosen_method,
            decision_reason=reason,
            final_verdict=verdict,
            merges_resolved=merges_count,
            spelling_corrections=spelling_count,
            tree_str=parse_res.tree_str,
        )

    def analyze_passage(
        self,
        processed_tokens: List[str],
        tagged_tokens: List[Tuple[str, str]],
        alerts: List[Any],
    ) -> Tuple[List[SentenceAnalysisRecord], pd.DataFrame]:
        """Analyze all sentences in the processed passage and construct the summary table."""
        sentences = self.split_into_sentences(processed_tokens)
        tagged_sentences = self.split_into_sentences([w for w, _ in tagged_tokens])

        records: List[SentenceAnalysisRecord] = []
        token_offset = 0

        for idx, sent_toks in enumerate(sentences):
            sent_len = len(sent_toks)
            sent_tagged = tagged_tokens[token_offset : token_offset + sent_len]

            # Count alerts belonging to this sentence span
            merges_in_sent = sum(
                1
                for a in alerts
                if a.alert_type == "[SEGMENT-ALERT]"
                and token_offset <= a.token_idx < token_offset + sent_len
            )
            spellings_in_sent = sum(
                1
                for a in alerts
                if a.alert_type == "[SPELL-ALERT]"
                and token_offset <= a.token_idx < token_offset + sent_len
            )

            rec = self.evaluate_sentence(
                sent_idx=idx,
                tokens=sent_toks,
                tagged_words=sent_tagged,
                merges_count=merges_in_sent,
                spelling_count=spellings_in_sent,
            )
            records.append(rec)
            token_offset += sent_len

        # Construct Pandas DataFrame for UI and report
        df = pd.DataFrame(
            [
                {
                    "Sentence #": r.sentence_idx,
                    "Sentence Text": r.sentence_text,
                    "PCFG Result": r.pcfg_result,
                    "Bigram Score (log2)": f"{r.bigram_score:.2f}",
                    "Bigram PPL": f"{r.bigram_ppl:.1f}",
                    "Trigram Score (log2)": f"{r.trigram_score:.2f}",
                    "Trigram PPL": f"{r.trigram_ppl:.1f}",
                    "Chosen Method": r.chosen_method,
                    "Final Verdict": r.final_verdict,
                    "Merges Resolved": r.merges_resolved,
                    "Spelling Fixed": r.spelling_corrections,
                }
                for r in records
            ]
        )

        return records, df
