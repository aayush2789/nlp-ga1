# NLP Group Assignment: Question 4
## Integrated Background Editor: Live Segmentation, Spelling Correction, and Constituency-Based Grammar Checking

This repository contains the complete implementation of **Question 4** for the NLP Group Assignment.

---

### Key Features
1. **Word Segmentation & POS Tagging**: Integrates Question 1's beam-search decoder and feature-based POS classifier with a modular adapter (`adapters/q1_adapter.py`) and development fallback mode.
2. **Spelling Correction**: Reuses Question 3's candidate generation (Method A: Edit-1, Method B: Symmetric Delete), unigram frequency non-word correction, and bigram contextual real-word error detection via `adapters/q3_adapter.py`.
3. **PCFG Constituency Parsing & Tagset Reconciliation**: Induces a Chomsky Normal Form PCFG from Penn Treebank, parsed via a probabilistic CKY/Viterbi parser with cell beam pruning ($K=25$) and Brown $\to$ Penn Treebank POS tagset reconciliation.
4. **Shared Smoothed N-Gram Language Models**: Add-$k$ ($k=0.05$) smoothed Bigram and Trigram models trained on the Brown corpus.
5. **Live Simulation & Interactive User Mode**: Fast-typing merge injection ($p=0.08$) and periodic grammar checks ($N=5$).
6. **Speed Demon Benchmark**: 1,000-word benchmark suite measuring latency and throughput, demonstrating a **17.1× speedup** of Method B over Method A and $0.074\text{ ms/word}$ per-token latency.
7. **Streamlit Web Application**: Interactive dashboard deployed via `app.py`.

---

### Project Structure
```
nlp-ga1/
├── config.py                       # Central hyperparameter configuration (p, N, k, thresholds)
├── data/
│   ├── corpora_loader.py           # Corpus loaders (Brown, Treebank, Gutenberg, Reuters)
│   └── tag_mappings.py             # Brown -> Penn Treebank POS tag reconciliation dictionary
├── adapters/
│   ├── q1_adapter.py               # Question 1 segmentation & POS decoder adapter (+ fallback)
│   └── q3_adapter.py               # Question 3 spelling correction adapter (+ fallback)
├── models/
│   ├── language_models.py          # Add-k smoothed Bigram & Trigram LMs (Brown corpus)
│   ├── pcfg_parser.py              # PCFG induction + optimized probabilistic CKY parser
│   └── tagset_reconciler.py        # Tag reconciliation layer between Q1 and PCFG
├── pipeline/
│   ├── typing_simulation.py        # Fast-typing merge generator (prob p), streamer
│   ├── live_processor.py           # Live token processor ([SEGMENT], [SPELL], [GRAMMAR] alerts)
│   └── final_analysis.py           # Sentence splitter, PCFG & N-gram scoring, decision rule
├── benchmark/
│   ├── speed_demon.py              # 1,000-word Speed Demon benchmark
│   └── comparative_analysis.py     # Subsystem interaction tracker, agreement matrix, sample runs
├── app.py                          # Streamlit interactive web application
├── run_benchmark.py                # Standalone CLI runner for Speed Demon and sample runs
├── tests/
│   └── test_pipeline.py            # Automated unit tests
└── report/
    ├── comparative_report.md       # Comprehensive technical & comparative analysis report
    └── sample_runs/                # Serialized JSON artifacts for Run 1, Run 2, and Speed Demon
```

---

### Quickstart

#### 1. Install Dependencies
```bash
pip install streamlit nltk pandas scikit-learn
python -c "import nltk; [nltk.download(c) for c in ['brown', 'treebank', 'gutenberg', 'reuters', 'punkt', 'punkt_tab', 'universal_tagset']]"
```

#### 2. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

#### 3. Run the Speed Demon Benchmark & Sample Runs
```bash
python run_benchmark.py
```

#### 4. Run Automated Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
