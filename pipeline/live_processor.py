"""Live Token Stream Processor with Segmentation, Spelling, and Grammar Alerting.

Implements the strict per-token pipeline:
1. Segmentation Check ([SEGMENT-ALERT]) via Question 1 beam decoder.
2. Spelling Correction Check ([SPELL-ALERT]) via Question 3 candidate generation.
3. Periodic Triggered Grammar Check ([GRAMMAR-ALERT]) every N words (local LM perplexity & real-word error check).

Instruments and tracks per-token latency and per-trigger latency in milliseconds.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
from adapters.q1_adapter import default_q1_adapter
from adapters.q3_adapter import default_q3_adapter
from config import EditorConfig, default_config
from models.language_models import SharedLanguageModels


@dataclass
class LiveEditorAlert:
    """Represents a warning or correction alert raised by the live editor."""
    alert_type: str  # '[SEGMENT-ALERT]', '[SPELL-ALERT]', or '[GRAMMAR-ALERT]'
    token_idx: int
    original: str
    replacement: str
    details: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class LiveEditorProcessor:
    """Stateful background editor engine processing tokens in real time."""

    def __init__(self, config: Optional[EditorConfig] = None):
        self.config = config or default_config
        self.q1 = default_q1_adapter
        self.q3 = default_q3_adapter
        self.lm = SharedLanguageModels.get_instance(k=self.config.smoothing_k)

        # Editor State
        self.processed_tokens: List[str] = []
        self.tagged_tokens: List[Tuple[str, str]] = []
        self.alerts: List[LiveEditorAlert] = []

        # Latency instrumentation
        self.per_token_latencies: List[float] = []
        self.per_trigger_latencies: List[float] = []

        # Counters
        self.merges_resolved: int = 0
        self.spelling_corrections: int = 0
        self.real_word_alerts: int = 0
        self.grammar_anomaly_alerts: int = 0

    def reset(self) -> None:
        """Clear all editor buffers and metrics."""
        self.processed_tokens.clear()
        self.tagged_tokens.clear()
        self.alerts.clear()
        self.per_token_latencies.clear()
        self.per_trigger_latencies.clear()
        self.merges_resolved = 0
        self.spelling_corrections = 0
        self.real_word_alerts = 0
        self.grammar_anomaly_alerts = 0

    def process_token(
        self, token: str, global_idx: int
    ) -> Tuple[List[str], List[LiveEditorAlert]]:
        """Process a single incoming token through the live pipeline.

        Returns:
            Tuple of (final_sub_tokens, newly_fired_alerts).
        """
        token_start_time = time.perf_counter()
        new_alerts: List[LiveEditorAlert] = []

        clean_tok = token.strip()
        if not clean_tok:
            return [], []

        # ---------------------------------------------------------------------
        # Step 1: Word Segmentation Check [SEGMENT-ALERT]
        # ---------------------------------------------------------------------
        # Triggered if token is not known in lexicon OR exceeds suspicious word length
        is_known = self.q1.is_known_word(clean_tok)
        is_suspicious_len = len(clean_tok) >= self.config.suspicious_word_length

        candidate_words = [clean_tok]
        candidate_tags = [self.q1.get_pos_tag(clean_tok)]

        if not is_known or is_suspicious_len:
            split_words, tags, split_score, single_score = self.q1.segment_token(clean_tok)
            if len(split_words) >= 2 and split_score > single_score:
                candidate_words = split_words
                candidate_tags = tags
                self.merges_resolved += 1

                tagged_repr = " + ".join(f"'{w}' ({t})" for w, t in zip(split_words, tags))
                seg_alert = LiveEditorAlert(
                    alert_type="[SEGMENT-ALERT]",
                    token_idx=global_idx,
                    original=clean_tok,
                    replacement=" ".join(split_words),
                    details=f"Resolved merged token via Q1 TrigramSegmenter: {tagged_repr}",
                    latency_ms=(time.perf_counter() - token_start_time) * 1000.0,
                )
                new_alerts.append(seg_alert)
                self.alerts.append(seg_alert)

        # ---------------------------------------------------------------------
        # Step 2: Spelling Correction Check [SPELL-ALERT]
        # ---------------------------------------------------------------------
        # Evaluated for each candidate word that remains outside the vocabulary
        final_words: List[str] = []
        final_tags: List[str] = []

        for w, tag in zip(candidate_words, candidate_tags):
            w_clean = w.strip(",.?!;:\'\"")
            if w_clean.isalpha() and not self.q3.is_known_word(w_clean):
                corrected, method_name, spell_latency = self.q3.correct_nonword(
                    w, method=self.config.spelling_candidate_method
                )
                if corrected.lower() != w.lower():
                    self.spelling_corrections += 1
                    spell_alert = LiveEditorAlert(
                        alert_type="[SPELL-ALERT]",
                        token_idx=global_idx,
                        original=w,
                        replacement=corrected,
                        details=(
                            f"Non-word correction: '{w}' -> '{corrected}' "
                            f"using {method_name} (candidate latency: {spell_latency:.2f}ms)"
                        ),
                        latency_ms=spell_latency,
                    )
                    new_alerts.append(spell_alert)
                    self.alerts.append(spell_alert)
                    final_words.append(corrected)
                    final_tags.append(self.q1.get_pos_tag(corrected))
                else:
                    final_words.append(w)
                    final_tags.append(tag)
            else:
                final_words.append(w)
                final_tags.append(tag)

        # Record per-token latency (segmentation + spelling)
        token_latency = (time.perf_counter() - token_start_time) * 1000.0
        self.per_token_latencies.append(token_latency)

        # Append to live history
        self.processed_tokens.extend(final_words)
        self.tagged_tokens.extend(zip(final_words, final_tags))

        # ---------------------------------------------------------------------
        # Step 3: Periodic Triggered Grammar & Real-Word Check [GRAMMAR-ALERT]
        # ---------------------------------------------------------------------
        # Runs every N words on the accumulated local context window
        if len(self.processed_tokens) % self.config.grammar_trigger_interval_N == 0:
            trigger_start_time = time.perf_counter()

            # Window of last words to inspect
            win_size = min(len(self.processed_tokens), self.config.grammar_window_size)
            window_tokens = self.processed_tokens[-win_size:]
            window_start_idx = len(self.processed_tokens) - win_size

            # (A) Real-Word Error Detection via Q3 Bigram LM
            for rel_idx in range(len(window_tokens)):
                curr_target = window_tokens[rel_idx]
                if not curr_target.isalpha():
                    continue

                rwe_res = self.q3.detect_real_word_error(window_tokens, rel_idx)
                if rwe_res:
                    best_cand, orig_p, cand_p = rwe_res
                    ratio = cand_p / max(1e-12, orig_p)
                    self.real_word_alerts += 1

                    rwe_alert = LiveEditorAlert(
                        alert_type="[GRAMMAR-ALERT]",
                        token_idx=window_start_idx + rel_idx,
                        original=curr_target,
                        replacement=best_cand,
                        details=(
                            f"Real-word contextual error detected: '{curr_target}' -> '{best_cand}'. "
                            f"Contextual likelihood ratio: {ratio:.1f}x higher (P_cand={cand_p:.2e} vs P_orig={orig_p:.2e})"
                        ),
                        latency_ms=(time.perf_counter() - trigger_start_time) * 1000.0,
                    )
                    new_alerts.append(rwe_alert)
                    self.alerts.append(rwe_alert)
                    # Apply real-word correction to buffer
                    self.processed_tokens[window_start_idx + rel_idx] = best_cand
                    window_tokens[rel_idx] = best_cand
                    break  # Avoid cascading duplicate checks within same trigger

            # (B) Window-level Trigram Perplexity Grammar Anomaly Check
            ppl = self.lm.trigram_model.sentence_perplexity(window_tokens)
            if ppl > self.config.grammar_perplexity_threshold:
                self.grammar_anomaly_alerts += 1
                anomaly_alert = LiveEditorAlert(
                    alert_type="[GRAMMAR-ALERT]",
                    token_idx=len(self.processed_tokens) - 1,
                    original=" ".join(window_tokens),
                    replacement="[Review syntax/collocation]",
                    details=(
                        f"Local window perplexity ({ppl:.1f}) exceeds anomaly threshold "
                        f"({self.config.grammar_perplexity_threshold:.1f}). Possible syntax or phrasing irregularity."
                    ),
                    latency_ms=(time.perf_counter() - trigger_start_time) * 1000.0,
                )
                new_alerts.append(anomaly_alert)
                self.alerts.append(anomaly_alert)

            trigger_latency = (time.perf_counter() - trigger_start_time) * 1000.0
            self.per_trigger_latencies.append(trigger_latency)

        return final_words, new_alerts

    def get_latency_stats(self) -> Dict[str, float]:
        """Return average and max latency metrics."""
        avg_token_lat = (
            sum(self.per_token_latencies) / len(self.per_token_latencies)
            if self.per_token_latencies
            else 0.0
        )
        avg_trigger_lat = (
            sum(self.per_trigger_latencies) / len(self.per_trigger_latencies)
            if self.per_trigger_latencies
            else 0.0
        )
        return {
            "avg_per_token_ms": avg_token_lat,
            "max_per_token_ms": max(self.per_token_latencies, default=0.0),
            "avg_per_trigger_ms": avg_trigger_lat,
            "max_per_trigger_ms": max(self.per_trigger_latencies, default=0.0),
            "total_tokens_processed": len(self.processed_tokens),
            "total_alerts": len(self.alerts),
        }
