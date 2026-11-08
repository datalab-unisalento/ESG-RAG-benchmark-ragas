# %%
"""Revision analysis: all pairwise intervals, per-pillar results, intersection of valid items,
and response lengths. Standard library only; deterministic with --seed.

Writes outputs/repro/revision_analysis.json.
"""
import argparse
import csv
import glob
import json
import math
import random
from itertools import combinations
from pathlib import Path

JUDGES = {
    "qwen3": "qwen3_4b-instruct",
    "nemotron": "nemotron-3.5-lightning_30b",
    "cogito": "cogito_32b",
    "granite": "granite4.2_30b",
    "lfm2.5": "lfm2.5_24b",
}
MODELS = ["glm-4.7-flash_q4_K_M", "ministral-3_8b", "gemma3_4b", "qwen3_4b-instruct", "nemotron-3-nano_4b"]
METRICS = ["faithfulness", "answer_relevancy", "factual_correctness", "context_recall",
           "llm_context_precision_with_reference"]
PAIRWISE_METRICS = ["faithfulness", "answer_relevancy", "factual_correctness"]


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


def mean(xs):
    xs = [x for x in xs if is_number(x)]
    return sum(xs) / len(xs) if xs else None


def paired_bootstrap(a, b, rng, reps=5000):
    pairs = [(x, y) for x, y in zip(a, b) if is_number(x) and is_number(y)]
    n = len(pairs)
    diff = sum(x - y for x, y in pairs) / n
    boots = []
    for _ in range(reps):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(x - y for x, y in s) / n)
    boots.sort()
    p_gt = sum(b > 0 for b in boots) / reps
    p_lt = sum(b < 0 for b in boots) / reps
    return {"diff": diff, "ci95": [boots[int(0.025 * reps)], boots[int(0.975 * reps)]],
            "p_two_sided_bootstrap": min(1.0, 2 * min(p_gt, p_lt)), "n_paired": n}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def pillar_of(persona):
    return {"Environmental KPI Extractor": "E", "Social KPI Extractor": "S",
            "Governance KPI Extractor": "G"}.get(persona, "?")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-dir", type=Path, default=Path("outputs/repro"))
    parser.add_argument("--judges-dir", type=Path, default=Path("judges"),
                        help="Judge output folder, relative to --repro-dir.")
    parser.add_argument("--testset", type=Path, default=Path("outputs/testset/01_esg_testset.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    scores = {}
    for j, d in JUDGES.items():
        judge_dir = args.repro_dir / args.judges_dir / d
        chunks = glob.glob(str(judge_dir / "*" / "chunk_*.json"))
        if len(chunks) == len(MODELS) * 10:  # 100 items in chunks of 10, per model
            scores[j] = {m: {r: per_item(judge_dir, r, m) for r in MODELS} for m in METRICS}
    report = {"seed": args.seed, "judges": list(scores), "pairwise": {}, "intersection": {},
              "per_pillar": {}, "year_histogram": {}, "mean_response_chars": {}}

    # All pairwise intervals, Holm-corrected within each judge and metric
    for j in scores:
        for m in PAIRWISE_METRICS:
            results, labels = [], []
            for a, b in combinations(MODELS, 2):
                results.append(paired_bootstrap(scores[j][m][a], scores[j][m][b], rng))
                labels.append(f"{a} minus {b}")
            adj = holm([r["p_two_sided_bootstrap"] for r in results])
            for label, r, pa in zip(labels, results, adj):
                report["pairwise"][f"{j} | {m} | {label}"] = {**r, "p_holm": pa}

    # Intersection: items with a valid score for every model
    for j in scores:
        for m in METRICS:
            n_items = len(scores[j][m][MODELS[0]])
            valid = [i for i in range(n_items) if all(is_number(scores[j][m][r][i]) for r in MODELS)]
            report["intersection"][f"{j} | {m}"] = {
                "items_valid_for_all_models": len(valid),
                "means": {r: (sum(scores[j][m][r][i] for i in valid) / len(valid)) if valid else None
                          for r in MODELS},
            }

    # Per-pillar results from each item's persona (first 100 testset rows)
    personas = []
    with open(args.testset, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            personas.append(pillar_of(row["persona"]))
    personas = personas[:100]
    for j in scores:
        for m in PAIRWISE_METRICS:
            for r in MODELS:
                for p in "ESG":
                    vals = [v for v, pil in zip(scores[j][m][r], personas) if pil == p]
                    report["per_pillar"][f"{j} | {m} | {r} | {p}"] = {
                        "mean": mean(vals), "n_valid": sum(is_number(v) for v in vals), "n": len(vals)}

    # Year of each source report in the testset (the last underscore field of the filename)
    years = {}
    with open(args.testset, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = Path(row["document_source"]).stem.split("_")[-1]
            years[year] = years.get(year, 0) + 1
    report["year_histogram"] = dict(sorted(years.items()))

    # Mean response length in characters, per model (first 100 items)
    for r in MODELS:
        recs = json.load(open(args.repro_dir / "rag_responses" / f"rag_responses_{r}.json", encoding="utf-8"))
        report["mean_response_chars"][r] = sum(len(x["response"]) for x in recs) / len(recs)

    out = args.repro_dir / "revision_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out}; judges: {list(scores)}")


if __name__ == "__main__":
    main()
