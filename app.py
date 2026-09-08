"""Integrated Background Editor Streamlit Application.

Question 4: Live Segmentation, Spelling Correction, and Constituency-Based Grammar Checking.
"""

import json
import os
from pathlib import Path
import time
import pandas as pd
import streamlit as st

from adapters.q1_adapter import default_q1_adapter
from adapters.q3_adapter import default_q3_adapter
from benchmark.comparative_analysis import ComparativeAnalyzer
from benchmark.speed_demon import SpeedDemonBenchmark
from config import EditorConfig, default_config
from models.language_models import SharedLanguageModels
from models.pcfg_parser import PCFGManager
from models.tagset_reconciler import default_reconciler
from pipeline.final_analysis import FinalPassageAnalyzer
from pipeline.live_processor import LiveEditorProcessor
from pipeline.typing_simulation import TypingSimulator

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Integrated Background Editor (Q4)",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    .alert-card-segment {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .alert-card-spell {
        background-color: #fefce8;
        border-left: 4px solid #eab308;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .alert-card-grammar {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .badge-fallback {
        background-color: #f59e0b;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-real {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached Global Model Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Shared Language Models & PCFG Parser...")
def load_system_engines():
    lm = SharedLanguageModels.get_instance(k=default_config.smoothing_k)
    pcfg_mgr = PCFGManager.get_instance(sample_size=default_config.treebank_sample_size)
    analyzer = FinalPassageAnalyzer(default_config)
    benchmarker = SpeedDemonBenchmark(default_config)
    comp_analyzer = ComparativeAnalyzer(default_config)
    return lm, pcfg_mgr, analyzer, benchmarker, comp_analyzer


lm, pcfg_mgr, analyzer, benchmarker, comp_analyzer = load_system_engines()

# -----------------------------------------------------------------------------
# Sidebar: System Status & Parameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ System Control & Status")

    # Q1 & Q3 Adapter Badges
    q1_status = default_q1_adapter.get_status()
    q3_status = default_q3_adapter.get_status()

    st.markdown("**Subsystem Integration Status:**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if q1_status["is_fallback"]:
            st.markdown("Q1: <span class='badge-fallback'>Fallback (Dev)</span>", unsafe_allow_html=True)
        else:
            st.markdown("Q1: <span class='badge-real'>Real (Loaded)</span>", unsafe_allow_html=True)
    with col_s2:
        if q3_status["is_fallback"]:
            st.markdown("Q3: <span class='badge-fallback'>Fallback (Dev)</span>", unsafe_allow_html=True)
        else:
            st.markdown("Q3: <span class='badge-real'>Real (Loaded)</span>", unsafe_allow_html=True)

    if q1_status.get("is_fallback", False):
        st.warning("⚠️ Q1 Fallback mode active.")
    elif q3_status.get("is_fallback", False):
        st.info("ℹ️ Q1 Real Model active (TrigramSegmenter + TrigramPOSTagger). Q3 running in standalone mode.")

    st.markdown("---")
    st.markdown("#### ⚙️ Tunable Hyperparameters")
    p_slider = st.slider(
        "Merge Probability (p)", min_value=0.0, max_value=0.30, value=0.08, step=0.01,
        help="Probability of dropping spaces between words to mimic fast typing."
    )
    n_slider = st.slider(
        "Grammar Trigger Interval (N)", min_value=2, max_value=15, value=5, step=1,
        help="Run grammar & real-word error checks every N words."
    )
    delay_slider = st.slider(
        "Simulated Typing Delay (sec)", min_value=0.01, max_value=0.40, value=0.08, step=0.01,
        help="Simulated inter-token typing speed."
    )
    k_slider = st.slider(
        "Add-k Smoothing (k)", min_value=0.01, max_value=0.50, value=0.05, step=0.01,
        help="Smoothing parameter for Bigram and Trigram models."
    )

    active_config = EditorConfig(
        merge_probability_p=p_slider,
        grammar_trigger_interval_N=n_slider,
        typing_delay_sec=delay_slider,
        smoothing_k=k_slider,
    )

# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>Integrated Background Editor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Question 4: Live Word Segmentation, Spelling Correction, and PCFG Constituency Grammar Checking</div>",
    unsafe_allow_html=True,
)

# Multi-tab layout
tabs = st.tabs([
    "🚀 Simulated Live Typing",
    "✍️ Interactive Live User Input",
    "📊 Sentence Analysis & Trees",
    "⚡ Speed Demon Benchmark",
    "🔬 Comparative Analysis & Interactions",
    "⚙️ Model & Adapter Inspector",
])

# =============================================================================
# TAB 1: Simulated Live Typing Demonstration
# =============================================================================
with tabs[0]:
    st.subheader("Simulated Human Typing Stream")
    st.caption("Streams a randomly sampled passage word-by-word, injecting fast-typing merge errors with probability p and raising live alerts in real time.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1.5, 1.5])
    with col_ctrl1:
        corpus_choice = st.selectbox("Sample Corpus:", ["gutenberg", "brown", "reuters"])
    with col_ctrl2:
        custom_seed = st.number_input("Random Seed (Optional):", min_value=1, max_value=99999, value=101)
    with col_ctrl3:
        start_btn = st.button("▶️ Start Live Simulation", type="primary", use_container_width=True)

    if start_btn:
        simulator = TypingSimulator(p=active_config.merge_probability_p, delay_sec=active_config.typing_delay_sec, seed=custom_seed)
        processor = LiveEditorProcessor(active_config)

        orig_passage, orig_sents, corrupted_tokens, injected_merges = simulator.prepare_passage(
            source=corpus_choice,
            min_sentences=active_config.min_sentences,
            max_sentences=active_config.max_sentences,
        )

        st.session_state["sim_orig_passage"] = orig_passage
        st.session_state["sim_corrupted_tokens"] = corrupted_tokens
        st.session_state["sim_injected_merges"] = injected_merges

        # Live Display Containers
        st.markdown("#### Live Editor Viewport")
        text_display = st.empty()
        alert_container = st.container()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        m_tokens = col_m1.empty()
        m_merges = col_m2.empty()
        m_spell = col_m3.empty()
        m_latency = col_m4.empty()

        all_streamed_words = []
        all_streamed_alerts = []

        # Stream loop
        progress_bar = st.progress(0.0)
        total_toks = len(corrupted_tokens)

        for idx, token in enumerate(corrupted_tokens):
            final_sub_toks, alerts = processor.process_token(token, idx)
            all_streamed_words.extend(final_sub_toks)
            all_streamed_alerts.extend(alerts)

            # Update live text display
            text_display.markdown(
                f"<div style='background-color: #f9fafb; padding: 16px; border: 1px solid #d1d5db; border-radius: 8px; font-family: monospace; font-size: 1.05rem; min-height: 100px;'>"
                f"{' '.join(all_streamed_words)} <span style='color: #4f46e5; font-weight: bold;'>|</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Update Metrics
            m_tokens.metric("Tokens Processed", len(all_streamed_words))
            m_merges.metric("Merges Resolved", processor.merges_resolved)
            m_spell.metric("Spelling Corrections", processor.spelling_corrections)
            lat_stats = processor.get_latency_stats()
            m_latency.metric("Avg Token Latency", f"{lat_stats['avg_per_token_ms']:.2f} ms")

            progress_bar.progress((idx + 1) / total_toks)
            if active_config.typing_delay_sec > 0:
                time.sleep(active_config.typing_delay_sec)

        st.success("✅ Passage streaming complete! Final sentence-level analysis ready in Tab 3.")

        # Save to session state for analysis tab
        st.session_state["sim_processor"] = processor
        st.session_state["sim_done"] = True

        # Display Alert Feed
        st.markdown("#### Live Alert Stream History")
        if processor.alerts:
            for a in reversed(processor.alerts):
                css_class = (
                    "alert-card-segment" if a.alert_type == "[SEGMENT-ALERT]"
                    else "alert-card-spell" if a.alert_type == "[SPELL-ALERT]"
                    else "alert-card-grammar"
                )
                st.markdown(
                    f"<div class='{css_class}'>"
                    f"<strong>{a.alert_type}</strong> @ token #{a.token_idx}: {a.details} "
                    f"<span style='color: #6b7280; font-size: 0.8rem;'>(latency: {a.latency_ms:.2f} ms)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No alerts triggered. All tokens passed validation cleanly.")

# =============================================================================
# TAB 2: Interactive Live User Input
# =============================================================================
with tabs[1]:
    st.subheader("Interactive Live Typing Mode")
    st.caption("Type or paste text directly. The pipeline processes text incrementally as words are typed, displaying real-time segmentation, spelling, and grammar alerts.")

    user_text = st.text_area(
        "Enter text to test:",
        value="The doctorexamined the patient closely. He hada high fever and needed medicine. I would like to sea the test results.",
        height=140,
    )

    if st.button("🔍 Process User Input", type="primary"):
        u_proc = LiveEditorProcessor(active_config)
        u_tokens = user_text.split()

        for idx, tok in enumerate(u_tokens):
            u_proc.process_token(tok, idx)

        col_u1, col_u2, col_u3, col_u4 = st.columns(4)
        col_u1.metric("Words Input", len(u_tokens))
        col_u2.metric("Merges Resolved", u_proc.merges_resolved)
        col_u3.metric("Spelling Corrections", u_proc.spelling_corrections)
        u_stats = u_proc.get_latency_stats()
        col_u4.metric("Avg Token Latency", f"{u_stats['avg_per_token_ms']:.2f} ms")

        st.markdown("#### Corrected Output Text")
        st.info(" ".join(u_proc.processed_tokens))

        st.markdown("#### Real-Time Alerts Fired")
        if u_proc.alerts:
            for a in u_proc.alerts:
                css_class = (
                    "alert-card-segment" if a.alert_type == "[SEGMENT-ALERT]"
                    else "alert-card-spell" if a.alert_type == "[SPELL-ALERT]"
                    else "alert-card-grammar"
                )
                st.markdown(
                    f"<div class='{css_class}'>"
                    f"<strong>{a.alert_type}</strong>: {a.details} "
                    f"<span style='color: #6b7280; font-size: 0.8rem;'>(latency: {a.latency_ms:.2f} ms)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("Clean sentence! No segmentation, spelling, or grammar issues detected.")

# =============================================================================
# TAB 3: Sentence-Level Analysis & PCFG Parse Trees
# =============================================================================
with tabs[2]:
    st.subheader("Part 4: Final Passage Analysis — Method Comparison")
    st.caption("Splits the corrected passage into sentences and scores each with PCFG parse log-prob, bigram log-prob, and trigram log-prob using the multi-tier decision rule.")

    if "sim_processor" in st.session_state:
        sim_proc = st.session_state["sim_processor"]
        records, df = analyzer.analyze_passage(
            sim_proc.processed_tokens, sim_proc.tagged_tokens, sim_proc.alerts
        )

        st.markdown("#### Sentence-Level Summary Table")
        st.dataframe(df, use_container_width=True)

        # CSV Download Button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Summary Table (CSV)",
            data=csv_data,
            file_name="q4_sentence_analysis.csv",
            mime="text/csv",
        )

        st.markdown("#### Interactive PCFG Constituency Parse Trees")
        for rec in records:
            with st.expander(f"Sentence #{rec.sentence_idx}: \"{rec.sentence_text}\" — Verdict: {rec.final_verdict} (via {rec.chosen_method})"):
                st.markdown(f"**Decision Rule Rationale:** {rec.decision_reason}")
                col_t1, col_t2, col_t3 = st.columns(3)
                col_t1.metric("PCFG Log-Prob", rec.pcfg_result)
                col_t2.metric("Bigram PPL", f"{rec.bigram_ppl:.1f}")
                col_t3.metric("Trigram PPL", f"{rec.trigram_ppl:.1f}")

                if rec.tree_str:
                    st.markdown("**Constituency Parse Tree Structure:**")
                    st.code(rec.tree_str, language="text")
                else:
                    st.warning("Constituency tree unparseable under current PCFG grammar productions.")
    else:
        st.info("Run a simulation in Tab 1 or process text in Tab 2 to view the final sentence analysis table.")

# =============================================================================
# TAB 4: Speed Demon Benchmark
# =============================================================================
with tabs[3]:
    st.subheader("Part 5: Speed Demon Benchmark (1,000 Words)")
    st.caption("Evaluates latency and throughput across the per-token pipeline (segmentation + spelling) versus isolated grammar checks, and compares Method A vs Method B candidate generation.")

    if st.button("⚡ Execute 1,000-Word Speed Demon Benchmark", type="primary"):
        with st.spinner("Generating 1,000 corrupted tokens and executing benchmarks..."):
            b_res = benchmarker.run_benchmark(batch_size=1000, seed=42)

            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            col_b1.metric("Per-Token Avg Latency", f"{b_res.per_token_avg_ms:.3f} ms/word")
            col_b2.metric("Per-Token Throughput", f"{b_res.per_token_throughput_wps:.0f} words/s")
            col_b3.metric("Method B Avg Latency", f"{b_res.method_b_avg_ms:.3f} ms/word")
            col_b4.metric("Method B vs A Speedup", f"{b_res.method_b_speedup:.1f}x")

            st.markdown("#### Latency Comparison Chart")
            latency_df = pd.DataFrame({
                "Component / Method": [
                    "Per-Token Check (Seg+Spell)",
                    "Isolated Grammar Trigger (Amortized)",
                    "Method A (Standard Edit-1)",
                    "Method B (Symmetric Delete)"
                ],
                "Average Latency (ms/word)": [
                    b_res.per_token_avg_ms,
                    b_res.grammar_avg_ms,
                    b_res.method_a_avg_ms,
                    b_res.method_b_avg_ms,
                ]
            })
            st.bar_chart(latency_df.set_index("Component / Method"))

            st.markdown("#### Analytical Benchmark Conclusion")
            st.info(b_res.conclusion)

# =============================================================================
# TAB 5: Comparative Analysis & Subsystem Interactions
# =============================================================================
with tabs[4]:
    st.subheader("Comparative Analysis & Subsystem Dynamics")
    st.caption("Examines agreement between real-time alerts and final sentence verdicts, structural vs. local error detection, and subsystem interaction dynamics.")

    st.markdown("#### Pre-Computed Full Demonstration Runs")
    run_choice = st.radio("Select Pre-Computed Sample Run to Inspect:", ["Run 1 (Gutenberg)", "Run 2 (Brown)"], horizontal=True)

    json_path = "report/sample_runs/run_1_gutenberg.json" if "Run 1" in run_choice else "report/sample_runs/run_2_brown.json"

    if Path(json_path).exists():
        with open(json_path, "r") as f:
            run_data = json.load(f)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Corpus Source", run_data["corpus_source"].upper())
        col_r2.metric("Sentences", run_data["sentence_count"])
        col_r3.metric("Merges Injected / Resolved", f"{len(run_data['injected_merges'])} / {run_data['merges_resolved']}")
        col_r4.metric("Spelling Fixes", run_data["spelling_corrections"])

        st.markdown("##### Real-Time Alert vs. Final Verdict Agreement Matrix")
        st.json(run_data["agreement_matrix"])

        st.markdown("##### Subsystem Interaction Effects (Pre- vs. Post-Correction)")
        if run_data["interactions"]:
            inter_df = pd.DataFrame(run_data["interactions"])
            st.dataframe(inter_df[["sentence_idx", "pcfg_before", "pcfg_after", "pcfg_flipped", "decision_before", "decision_after", "decision_changed"]], use_container_width=True)
        else:
            st.info("No sentence flips observed in this run.")

        st.markdown("##### Full Sentence Analysis Table from Run")
        st.dataframe(pd.DataFrame(run_data["sentence_records"]), use_container_width=True)
    else:
        st.warning(f"Run file '{json_path}' not found. Run `python run_benchmark.py` to generate it.")

# =============================================================================
# TAB 6: Model & Adapter Inspector
# =============================================================================
with tabs[5]:
    st.subheader("Subsystem Adapter & Configuration Inspector")
    st.caption("Inspect integration interfaces, active model instances, and POS tagset reconciliation mappings.")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("#### Question 1 Adapter Details")
        st.json(default_q1_adapter.get_status())

    with col_i2:
        st.markdown("#### Question 3 Adapter Details")
        st.json(default_q3_adapter.get_status())

    st.markdown("#### POS Tagset Reconciliation (Brown → Penn Treebank)")
    st.markdown("Question 1 uses Brown Corpus POS tags while the PCFG expects Penn Treebank tags. The reconciler uses normalized lookup with modifier stripping and prefix fallback:")
    recon_stats = default_reconciler.get_coverage_stats()
    st.json(recon_stats)
