"""CLI Runner for Speed Demon Benchmark and Sample Runs.

Executes:
1. Speed Demon 1,000-word latency & throughput benchmark.
2. Two full reproducible demonstration runs on different corpora.
3. Generates JSON artifacts for the comparative analysis report.
"""

import json
from pathlib import Path
from benchmark.comparative_analysis import ComparativeAnalyzer
from benchmark.speed_demon import SpeedDemonBenchmark
from config import EditorConfig


def main():
    print("=" * 80)
    print(" QUESTION 4: INTEGRATED BACKGROUND EDITOR BENCHMARK & DEMO RUNNER")
    print("=" * 80)

    cfg = EditorConfig(
        merge_probability_p=0.08,
        grammar_trigger_interval_N=5,
        smoothing_k=0.05,
        typing_delay_sec=0.0,
    )

    # 1. Run Speed Demon Benchmark
    print("\n>>> [1/3] Running Speed Demon Benchmark (1,000 corrupted words)...")
    benchmarker = SpeedDemonBenchmark(config=cfg)
    results = benchmarker.run_benchmark(batch_size=1000, seed=42)

    print("\n" + "-" * 60)
    print("SPEED DEMON BENCHMARK RESULTS")
    print("-" * 60)
    print(f"Batch Size:                      {results.batch_size} words")
    print(f"Full Per-Token Pipeline Time:    {results.per_token_total_sec:.3f} s")
    print(f"Per-Token Average Latency:       {results.per_token_avg_ms:.3f} ms/word")
    print(f"Per-Token Throughput:            {results.per_token_throughput_wps:.1f} words/sec")
    print(f"Isolated Grammar Check Time:     {results.grammar_total_sec:.3f} s")
    print(f"Grammar Amortized Latency:       {results.grammar_avg_ms:.3f} ms/word")
    print(f"Method A (Edit-1) Avg Latency:   {results.method_a_avg_ms:.3f} ms/word")
    print(f"Method B (SymDelete) Avg Latency:{results.method_b_avg_ms:.3f} ms/word")
    print(f"Method B vs. Method A Speedup:   {results.method_b_speedup:.1f}x")
    print("-" * 60)
    print("\nBenchmark Conclusion:")
    print(results.conclusion)

    # Save benchmark results to JSON
    bench_out = Path("report/sample_runs/speed_demon_results.json")
    bench_out.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_out, "w") as f:
        json.dump(
            {
                "batch_size": results.batch_size,
                "per_token_total_sec": results.per_token_total_sec,
                "per_token_avg_ms": results.per_token_avg_ms,
                "per_token_throughput_wps": results.per_token_throughput_wps,
                "grammar_total_sec": results.grammar_total_sec,
                "grammar_avg_ms": results.grammar_avg_ms,
                "grammar_throughput_wps": results.grammar_throughput_wps,
                "method_a_avg_ms": results.method_a_avg_ms,
                "method_b_avg_ms": results.method_b_avg_ms,
                "method_b_speedup": results.method_b_speedup,
                "conclusion": results.conclusion,
            },
            f,
            indent=2,
        )
    print(f"Saved benchmark results to {bench_out}")

    # 2. Run Sample Run 1 (Gutenberg)
    print("\n>>> [2/3] Executing Sample Run 1 (Gutenberg Corpus)...")
    analyzer = ComparativeAnalyzer(cfg)
    run_1 = analyzer.execute_sample_run(
        run_id="run_1_gutenberg", source="gutenberg", seed=2024
    )
    path_1 = analyzer.save_run_report(run_1)
    print(f"Sample Run 1 completed: {run_1.sentence_count} sentences processed.")
    print(f"Merges Injected: {len(run_1.injected_merges)}, Resolved: {run_1.merges_resolved}")
    print(f"Spelling Fixes: {run_1.spelling_corrections}, Alerts Raised: {len(run_1.alerts)}")
    print(f"Saved Run 1 record to: {path_1}")

    # 3. Run Sample Run 2 (Brown Corpus)
    print("\n>>> [3/3] Executing Sample Run 2 (Brown Corpus)...")
    run_2 = analyzer.execute_sample_run(
        run_id="run_2_brown", source="brown", seed=999
    )
    path_2 = analyzer.save_run_report(run_2)
    print(f"Sample Run 2 completed: {run_2.sentence_count} sentences processed.")
    print(f"Merges Injected: {len(run_2.injected_merges)}, Resolved: {run_2.merges_resolved}")
    print(f"Spelling Fixes: {run_2.spelling_corrections}, Alerts Raised: {len(run_2.alerts)}")
    print(f"Saved Run 2 record to: {path_2}")

    print("\n" + "=" * 80)
    print(" ALL BENCHMARKS AND DEMONSTRATION RUNS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
