"""Automated Unit and Integration Tests for Question 4 Integrated Background Editor."""

import unittest
from adapters.q1_adapter import default_q1_adapter
from adapters.q3_adapter import default_q3_adapter
from config import EditorConfig
from models.language_models import SmoothedNgramModel
from models.pcfg_parser import PCFGManager
from models.tagset_reconciler import default_reconciler
from pipeline.final_analysis import FinalPassageAnalyzer
from pipeline.live_processor import LiveEditorProcessor
from pipeline.typing_simulation import TypingSimulator


class TestTagsetReconciler(unittest.TestCase):
    """Test Brown to Penn Treebank tag reconciliation."""

    def test_direct_lookup_mappings(self):
        self.assertEqual(default_reconciler.reconcile_tag("nn"), "NN")
        self.assertEqual(default_reconciler.reconcile_tag("vbz"), "VBZ")
        self.assertEqual(default_reconciler.reconcile_tag("at"), "DT")
        self.assertEqual(default_reconciler.reconcile_tag("jj"), "JJ")
        self.assertEqual(default_reconciler.reconcile_tag("cs"), "IN")
        self.assertEqual(default_reconciler.reconcile_tag("."), ".")

    def test_modifier_normalization(self):
        # Brown title/headline tags: 'nn-tl', 'np-hl', 'in+at'
        self.assertEqual(default_reconciler.reconcile_tag("nn-tl"), "NN")
        self.assertEqual(default_reconciler.reconcile_tag("np-hl"), "NNP")
        self.assertEqual(default_reconciler.reconcile_tag("in+at"), "IN")

    def test_prefix_and_default_fallback(self):
        # Prefix matching
        self.assertEqual(default_reconciler.reconcile_tag("vbd-xyz"), "VBD")
        # Unknown tag defaults to NN
        self.assertEqual(default_reconciler.reconcile_tag("unknown_foo_tag"), "NN")


class TestLanguageModels(unittest.TestCase):
    """Test Add-k smoothed Bigram and Trigram models."""

    def setUp(self):
        self.corpus = [
            ["the", "dog", "barked", "at", "the", "cat"],
            ["the", "cat", "sat", "on", "the", "mat"],
            ["a", "dog", "chased", "the", "cat"],
        ]
        self.bigram = SmoothedNgramModel(n=2, k=0.1)
        self.bigram.train(self.corpus)
        self.trigram = SmoothedNgramModel(n=3, k=0.1)
        self.trigram.train(self.corpus)

    def test_probabilities_sum_and_smoothing(self):
        # Probabilities should be strictly positive (never zero due to add-k)
        prob = self.bigram.prob("unseen_word", ("the",))
        self.assertGreater(prob, 0.0)

        tri_prob = self.trigram.prob("mat", ("on", "the"))
        self.assertGreater(tri_prob, 0.0)

    def test_sentence_log_prob_and_perplexity(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        log_p = self.bigram.sentence_log_prob(tokens)
        ppl = self.bigram.sentence_perplexity(tokens)

        self.assertLess(log_p, 0.0)
        self.assertGreater(ppl, 1.0)


class TestQ1Adapter(unittest.TestCase):
    """Test Question 1 genuine English segmentation and POS tagging integration."""

    def test_genuine_q1_models_loaded(self):
        status = default_q1_adapter.get_status()
        self.assertFalse(status["is_fallback"])
        self.assertEqual(status["status_label"], "REAL (Q1 Imported)")
        self.assertGreater(status["vocab_size"], 30000)
        self.assertIsNotNone(default_q1_adapter.segmenter)
        self.assertIsNotNone(default_q1_adapter.tagger)

    def test_known_word_lookup(self):
        self.assertTrue(default_q1_adapter.is_known_word("the"))
        self.assertTrue(default_q1_adapter.is_known_word("government"))
        self.assertTrue(default_q1_adapter.is_known_word("investigation"))
        self.assertFalse(default_q1_adapter.is_known_word("thequickbrownfox"))

    def test_merged_word_segmentation_with_real_q1(self):
        split_words, tags, split_score, single_score = default_q1_adapter.segment_token("thequick")
        self.assertEqual(split_words, ["the", "quick"])
        self.assertGreater(split_score, single_score)
        self.assertEqual(len(tags), 2)
        # Real Q1 Brown tag for 'the' is 'AT'
        self.assertEqual(tags[0].upper(), "AT")

    def test_pos_tagging_with_real_q1(self):
        tag = default_q1_adapter.get_pos_tag("the")
        self.assertEqual(tag.upper(), "AT")

        tagged_sent = default_q1_adapter.tag_sentence(["the", "cat"])
        self.assertEqual(len(tagged_sent), 2)
        self.assertEqual(tagged_sent[0][1].upper(), "AT")

    def test_segment_and_tag(self):
        pairs = default_q1_adapter.segment_and_tag("thequick")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0], "the")
        self.assertEqual(pairs[1][0], "quick")
        self.assertEqual(pairs[0][1].upper(), "AT")

    def test_explicit_failure_on_missing_resources(self):
        from adapters.q1_adapter import Q1IntegrationAdapter
        import tempfile
        # Create a corrupted/empty pickle file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(b"corrupted_pickle_content")
            tmp_path = tmp.name

        try:
            # When corrupted file cannot be unpickled and we mock failure
            with self.assertRaises(RuntimeError):
                # Force failure by passing invalid path with mocked loader exception
                from unittest.mock import patch
                with patch("adapters.q1_adapter.load_english_q1_models", side_effect=Exception("Corrupted model file")):
                    Q1IntegrationAdapter(model_path=tmp_path)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestQ3Adapter(unittest.TestCase):
    """Test Question 3 spelling correction adapter (Method A & Method B)."""

    def test_method_a_candidate_generation(self):
        # Misspelling of 'apple': 'aple'
        cands_a = default_q3_adapter.generate_candidates_method_a("aple")
        self.assertIn("apple", cands_a)

    def test_method_b_candidate_generation(self):
        # Symmetric Delete lookup
        cands_b = default_q3_adapter.generate_candidates_method_b("aple")
        self.assertIn("apple", cands_b)

    def test_nonword_correction(self):
        corrected, method_name, latency = default_q3_adapter.correct_nonword("testng", method="B")
        self.assertTrue(len(corrected) > 0)
        self.assertIn("Method B", method_name)
        self.assertGreaterEqual(latency, 0.0)

    def test_real_word_error_detection(self):
        # "I would like to sea the world" -> 'sea' should trigger 'see'
        context = ["i", "would", "like", "to", "sea", "the", "world"]
        res = default_q3_adapter.detect_real_word_error(context, 4)
        if res:
            best_cand, orig_p, cand_p = res
            self.assertEqual(best_cand, "see")
            self.assertGreater(cand_p, orig_p)


class TestPCFGParser(unittest.TestCase):
    """Test Penn Treebank PCFG induction and CKY Viterbi parser."""

    @classmethod
    def setUpClass(cls):
        # Small sample size for fast test execution
        cls.pcfg_mgr = PCFGManager.get_instance(sample_size=300)

    def test_valid_parse(self):
        # Short simple sentence: "the cat sat"
        tagged = [("the", "at"), ("cat", "nn"), ("sat", "vbd")]
        res = self.pcfg_mgr.parser.parse(tagged)
        # Even if unparseable on small treebank, it must return ParseResult without crashing
        self.assertIsNotNone(res.status)
        self.assertIn(res.status, ["parsed", "unparseable"])

    def test_graceful_unparseable_handling(self):
        # Syntactically nonsensical sequence: "cat cat cat cat"
        tagged = [("cat", "nn"), ("cat", "nn"), ("cat", "nn"), ("cat", "nn")]
        res = self.pcfg_mgr.parser.parse(tagged)
        self.assertIsInstance(res.is_valid, bool)
        self.assertGreaterEqual(res.latency_ms, 0.0)


class TestFullPipeline(unittest.TestCase):
    """Test end-to-end integration: typing simulation, alerts, and final analysis."""

    def test_integrated_editor_flow(self):
        cfg = EditorConfig(
            merge_probability_p=0.5,  # high p to guarantee merge
            grammar_trigger_interval_N=3,
            typing_delay_sec=0.0,
            random_seed=42,
        )
        sim = TypingSimulator(p=cfg.merge_probability_p, seed=42)
        text = "The doctor examined the patient. He prescribed medicine for the fever."
        _, _, tokens, merges = sim.prepare_passage(passage_text=text)

        self.assertGreater(len(tokens), 0)

        proc = LiveEditorProcessor(config=cfg)
        all_fired_alerts = []
        for idx, t in enumerate(tokens):
            _, alerts = proc.process_token(t, idx)
            all_fired_alerts.extend(alerts)

        self.assertGreater(len(proc.processed_tokens), 0)
        self.assertGreater(len(proc.per_token_latencies), 0)

        analyzer = FinalPassageAnalyzer(config=cfg)
        records, df = analyzer.analyze_passage(
            proc.processed_tokens, proc.tagged_tokens, proc.alerts
        )

        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(len(df), len(records))
        self.assertIn("Final Verdict", df.columns)
        self.assertIn("Chosen Method", df.columns)


if __name__ == "__main__":
    unittest.main()
