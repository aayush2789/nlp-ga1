"""Comparative Analysis and Subsystem Interaction Tracker.

Tracks and calculates:
1. Real-time alert vs. final sentence verdict agreement / disagreement matrices
2. Structural (PCFG) vs. Local Sequence (N-gram) error detection divergence
3. Subsystem interaction effects (e.g. spelling or segmentation fixes flipping
   PCFG parseability from unparseable -> parsed or altering the chosen decision rule)
4. Execution and serialization of at least two complete reproducible sample runs.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import EditorConfig, default_config
from pipeline.final_analysis import FinalPassageAnalyzer, SentenceAnalysisRecord
from pipeline.live_processor import LiveEditorProcessor
from pipeline.typing_simulation import TypingSimulator


@dataclass
class SubsystemInteraction:
    """Records an interaction where an earlier correction altered a downstream model's behavior."""
    sentence_idx: int
    raw_sentence: str
    corrected_sentence: str
    pcfg_before: str
    pcfg_after: str
    pcfg_flipped: bool  # True if unparseable -> parsed
    decision_before: str
    decision_after: str
    decision_changed: bool
    corrections_applied: List[str]


@dataclass
class SampleRunReport:
    """Complete record of a single end-to-end demonstration run."""
    run_id: str
    corpus_source: str
    passage_text: str
    sentence_count: int
    injected_merges: List[Dict[str, Any]]
    total_tokens_processed: int
    merges_resolved: int
    spelling_corrections: int
    real_word_alerts: int
    grammar_anomalies: int
    alerts: List[Dict[str, Any]]
    latency_stats: Dict[str, float]
    sentence_records: List[Dict[str, Any]]
    interactions: List[Dict[str, Any]]
    agreement_matrix: Dict[str, int]


class ComparativeAnalyzer:
    """Evaluates cross-system dynamics, interaction effects, and generates sample run artifacts."""

    def __init__(self, config: Optional[EditorConfig] = None):
        self.config = config or default_config
        self.simulator = TypingSimulator(
            p=self.config.merge_probability_p,
            delay_sec=0.0,  # zero delay for analytical batch execution
        )
        self.processor = LiveEditorProcessor(self.config)
        self.analyzer = FinalPassageAnalyzer(self.config)

    def execute_sample_run(
        self,
        run_id: str = "run_1",
        source: str = "gutenberg",
        seed: Optional[int] = 101,
        custom_text: Optional[str] = None,
    ) -> SampleRunReport:
        """Execute a full demonstration run and gather all telemetry."""
        self.simulator.seed = seed
        self.simulator.rng.seed(seed)
        self.processor.reset()

        # 1. Prepare passage & inject fast-typing merges
        orig_passage, orig_sents, corrupted_tokens, injected_merges = (
            self.simulator.prepare_passage(
                passage_text=custom_text,
                source=source,
                min_sentences=self.config.min_sentences,
                max_sentences=self.config.max_sentences,
            )
        )

        # 2. Process token-by-token through live pipeline
        for idx, token in enumerate(corrupted_tokens):
            self.processor.process_token(token, idx)

        # 3. Perform final passage analysis
        records, _ = self.analyzer.analyze_passage(
            self.processor.processed_tokens,
            self.processor.tagged_tokens,
            self.processor.alerts,
        )

        # 4. Measure Subsystem Interaction Effects (Pre- vs Post-Correction)
        interactions = self._detect_interactions(orig_sents, records)

        # 5. Compute Agreement Matrix (Real-time alerts vs Final Verdict)
        agreement = self._compute_agreement(records, self.processor.alerts)

        report = SampleRunReport(
            run_id=run_id,
            corpus_source=source,
            passage_text=orig_passage,
            sentence_count=len(records),
            injected_merges=[
                {
                    "token_idx": m.token_idx,
                    "merged_token": m.merged_token,
                    "original_words": m.original_words,
                    "sentence_idx": m.sentence_idx,
                }
                for m in injected_merges
            ],
            total_tokens_processed=len(self.processor.processed_tokens),
            merges_resolved=self.processor.merges_resolved,
            spelling_corrections=self.processor.spelling_corrections,
            real_word_alerts=self.processor.real_word_alerts,
            grammar_anomalies=self.processor.grammar_anomaly_alerts,
            alerts=[
                {
                    "alert_type": a.alert_type,
                    "token_idx": a.token_idx,
                    "original": a.original,
                    "replacement": a.replacement,
                    "details": a.details,
                    "latency_ms": a.latency_ms,
                }
                for a in self.processor.alerts
            ],
            latency_stats=self.processor.get_latency_stats(),
            sentence_records=[
                {
                    "sentence_idx": r.sentence_idx,
                    "sentence_text": r.sentence_text,
                    "pcfg_result": r.pcfg_result,
                    "pcfg_log_prob": r.pcfg_log_prob,
                    "bigram_score": r.bigram_score,
                    "bigram_ppl": r.bigram_ppl,
                    "trigram_score": r.trigram_score,
                    "trigram_ppl": r.trigram_ppl,
                    "chosen_method": r.chosen_method,
                    "decision_reason": r.decision_reason,
                    "final_verdict": r.final_verdict,
                    "merges_resolved": r.merges_resolved,
                    "spelling_corrections": r.spelling_corrections,
                    "tree_str": r.tree_str,
                }
                for r in records
            ],
            interactions=[asdict(item) for item in interactions],
            agreement_matrix=agreement,
        )

        return report

    def _detect_interactions(
        self, original_sents: List[str], records: List[SentenceAnalysisRecord]
    ) -> List[SubsystemInteraction]:
        """Compare uncorrected vs corrected sentences to observe interaction dynamics."""
        interactions: List[SubsystemInteraction] = []

        for idx, rec in enumerate(records):
            if idx >= len(original_sents):
                break

            orig_str = original_sents[idx]
            corr_str = rec.sentence_text

            if orig_str.strip().lower() != corr_str.strip().lower():
                # Evaluate raw uncorrected sentence with PCFG
                raw_words = orig_str.split()
                raw_tagged = self.processor.q1.tag_sentence(raw_words)
                raw_parse = self.analyzer.pcfg_mgr.parser.parse(
                    raw_tagged, max_sentence_len=self.config.pcfg_max_sentence_length
                )

                pcfg_before = f"{raw_parse.log_prob:.2f}" if raw_parse.is_valid else "unparseable"
                pcfg_after = rec.pcfg_result
                flipped = (not raw_parse.is_valid) and (rec.pcfg_result != "unparseable")

                # Decision before vs after
                dec_before = "PCFG Parser" if raw_parse.is_valid else "Trigram LM"
                dec_after = rec.chosen_method

                interactions.append(
                    SubsystemInteraction(
                        sentence_idx=rec.sentence_idx,
                        raw_sentence=orig_str,
                        corrected_sentence=corr_str,
                        pcfg_before=pcfg_before,
                        pcfg_after=pcfg_after,
                        pcfg_flipped=flipped,
                        decision_before=dec_before,
                        decision_after=dec_after,
                        decision_changed=(dec_before != dec_after),
                        corrections_applied=[
                            f"Merges resolved: {rec.merges_resolved}",
                            f"Spelling fixes: {rec.spelling_corrections}",
                        ],
                    )
                )

        return interactions

    def _compute_agreement(
        self, records: List[SentenceAnalysisRecord], alerts: List[Any]
    ) -> Dict[str, int]:
        """Compute agreement/disagreement frequency between live alerts and final verdicts."""
        agreement = {
            "alerts_raised_and_verdict_ungrammatical_or_questionable": 0,
            "alerts_raised_but_verdict_grammatical_post_correction": 0,
            "no_alerts_and_verdict_grammatical": 0,
            "no_alerts_but_verdict_ungrammatical_or_questionable": 0,
        }

        for rec in records:
            has_alerts = (rec.merges_resolved > 0) or (rec.spelling_corrections > 0)
            is_grammatical = rec.final_verdict == "Grammatical"

            if has_alerts and not is_grammatical:
                agreement["alerts_raised_and_verdict_ungrammatical_or_questionable"] += 1
            elif has_alerts and is_grammatical:
                # Common occurrence: live alerts successfully repaired the sentence,
                # allowing the final corrected sentence to parse cleanly as Grammatical!
                agreement["alerts_raised_but_verdict_grammatical_post_correction"] += 1
            elif not has_alerts and is_grammatical:
                agreement["no_alerts_and_verdict_grammatical"] += 1
            else:
                # Structural syntax error caught by PCFG/LM without lexical misspelling
                agreement["no_alerts_but_verdict_ungrammatical_or_questionable"] += 1

        return agreement

    def save_run_report(self, report: SampleRunReport, output_dir: str = "report/sample_runs") -> str:
        """Serialize a sample run to JSON format."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{report.run_id}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

        return str(file_path)
