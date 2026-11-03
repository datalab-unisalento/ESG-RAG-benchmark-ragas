# %%
"""Summarize the controlled reproduction: judge agreement and paired bootstrap intervals.

Reads per-item factual-correctness scores from outputs/repro/judges and writes
outputs/repro/analysis.json. Standard library only; deterministic with --seed.
"""
import argparse
import glob
import json
import math
import random
from pathlib import Path

JUDGES = {
    "qwen3": "qwen3_4b-instruct",
    "nemotron": "nemotron-3.5-lightning_30b",
    "cogito": "cogito_32b",
}
MODELS = ["gemma3_4b", "glm-4.7-flash_q4_K_M", "ministral-3_8b", "nemotron-3-nano_4b", "qwen3_4b-instruct"]
METRIC = "factual_correctness"


def is_number(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def per_item_scores(judge_dir, rag, metric):
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


def average_ranks(xs):
    # Ties receive the mean of the ranks they span
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    # Pearson correlation of average ranks (tie-corrected Spearman)
    ra, rb = average_ranks(a), average_ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb)


def paired_bootstrap(a, b, rng, reps=5000):
    pairs = [(x, y) for x, y in zip(a, b) if is_number(x) and is_number(y)]
    n = len(pairs)
    diff = sum(x - y for x, y in pairs) / n
    boots = []
    for _ in range(reps):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(x - y for x, y in s) / n)
    boots.sort()
    return {
        "diff": diff,
        "ci95": [boots[int(0.025 * reps)], boots[int(0.975 * reps)]],
        "p_gt_0": sum(b > 0 for b in boots) / reps,
        "n_paired": n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-dir", type=Path, default=Path("outputs/repro"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fc = {j: {m: per_item_scores(args.repro_dir / "judges" / d, m, METRIC) for m in MODELS}
          for j, d in JUDGES.items()}
    report = {"seed": args.seed, "metric": METRIC, "judges": {}, "spearman_over_5_models": {}, "pairwise": {}}

    for j in JUDGES:
        report["judges"][j] = {
            "means": {m: mean(fc[j][m]) for m in MODELS},
            "missing": {m: sum(not is_number(x) for x in fc[j][m]) for m in MODELS},
        }

    means = {j: [mean(fc[j][m]) for m in MODELS] for j in JUDGES}
    report["spearman_over_5_models"]["qwen3_vs_nemotron"] = spearman(means["qwen3"], means["nemotron"])
    report["spearman_over_5_models"]["qwen3_vs_cogito"] = spearman(means["qwen3"], means["cogito"])
    report["spearman_over_5_models"]["nemotron_vs_cogito"] = spearman(means["nemotron"], means["cogito"])

    for j in JUDGES:
        report["pairwise"][f"{j}: qwen3 minus gemma3"] = paired_bootstrap(
            fc[j]["qwen3_4b-instruct"], fc[j]["gemma3_4b"], rng)

    out = args.repro_dir / "analysis.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
