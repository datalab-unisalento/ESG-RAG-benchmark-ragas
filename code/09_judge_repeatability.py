# %%
"""Compare two scorings of the same responses by the same judge (repeatability check).

Reads per-item scores from two judge output directories (as written by
04_compare_ragas_judges.py) and reports, per model and metric, how often the two
runs agree exactly and by how much they differ. Writes outputs/repro/repeatability.json.
"""
import argparse
import glob
import json
import math
from pathlib import Path

MODELS = ["gemma3_4b", "glm-4.7-flash_q4_K_M", "ministral-3_8b", "nemotron-3-nano_4b", "qwen3_4b-instruct"]
METRICS = ["llm_context_precision_with_reference", "context_precision", "context_recall",
           "context_relevance", "faithfulness", "factual_correctness", "answer_relevancy"]


def is_number(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def per_item(judge_dir, rag, metric):
    values = []
    chunks = sorted(glob.glob(f"{judge_dir}/{rag}/chunk_*.json"),
                    key=lambda p: int(Path(p).stem.split("_")[1]))
    for f in chunks:
        for s in json.load(open(f))["scores"]:
            values.append(s.get(metric))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, default=Path("outputs/repro/judges/qwen3_4b-instruct"))
    parser.add_argument("--run-b", type=Path, default=Path("outputs/repro/judges_repeat/qwen3_4b-instruct"))
    parser.add_argument("--output", type=Path, default=Path("outputs/repro/repeatability.json"))
    args = parser.parse_args()

    report = {"run_a": str(args.run_a), "run_b": str(args.run_b), "results": {}}
    for model in MODELS:
        report["results"][model] = {}
        for metric in METRICS:
            a = per_item(args.run_a, model, metric)
            b = per_item(args.run_b, model, metric)
            pairs = [(x, y) for x, y in zip(a, b) if is_number(x) and is_number(y)]
            diffs = [abs(x - y) for x, y in pairs]
            report["results"][model][metric] = {
                "paired_items": len(pairs),
                "exact_matches": sum(d == 0 for d in diffs),
                "max_abs_diff": max(diffs) if diffs else None,
                "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
                "missing_a": sum(not is_number(x) for x in a),
                "missing_b": sum(not is_number(x) for x in b),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    for model in MODELS:
        for metric in METRICS:
            r = report["results"][model][metric]
            print(f"{model:24s} {metric:38s} paired={r['paired_items']:3d} exact={r['exact_matches']:3d} "
                  f"max={r['max_abs_diff']} mean={r['mean_abs_diff']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
