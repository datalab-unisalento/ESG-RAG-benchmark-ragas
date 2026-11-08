# %%
"""Measure generation token counts and latency per model on saved RAG prompts.

Uses the same fixed decoding and context settings as 02_run_rag_generation.py.
Each model is loaded alone; its results are saved on completion and skipped on resume.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from langchain_ollama import ChatOllama
import pandas as pd

# rag_responses file suffix -> exact Ollama tag
MODEL_TAGS = {
    "gemma3_4b": "gemma3:4b",
    "glm-4.7-flash_q4_K_M": "glm-4.7-flash:q4_K_M",
    "ministral-3_8b": "ministral-3:8b",
    "nemotron-3-nano_4b": "nemotron-3-nano:4b",
    "qwen3_4b-instruct": "qwen3:4b-instruct",
}

SYSTEM = "You are a helpful assistant that answers questions based on given documents only."
GEN_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024,
               "reasoning": False}  # thinking disabled: hidden reasoning otherwise uses the token budget


def build_messages(query, context):
    # Same prompt as generate_answer() in 02_run_rag_generation.py
    prompt = f"question: {query}\n\nDocuments: {context}"
    return [("system", SYSTEM), ("human", prompt)]


def unload(tag):
    # Unload the model so the next one does not compete for GPU memory
    subprocess.run(["ollama", "stop", tag], check=False)


def placement():
    # Record where Ollama currently runs the loaded model (GPU vs CPU split)
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=False).stdout
    return out.strip()


def ns_to_s(value):
    return value / 1e9 if value is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-dir", type=Path, default=Path("outputs/repro/rag_responses"))
    parser.add_argument("--n", type=int, default=10, help="Number of queries per model.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/repro/benchmark"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    per_model_dir = args.output_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"rag_dir": str(args.rag_dir), "n": args.n, "generation_options": GEN_OPTIONS,
                "models": MODEL_TAGS}
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            sys.exit(f"{manifest_path} differs from this configuration; use a new --output-dir.")
        if not args.resume:
            sys.exit(f"{args.output_dir} already has a run; pass --resume.")
    else:
        if args.resume:
            sys.exit("--resume given but no manifest exists.")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for suffix, tag in MODEL_TAGS.items():
        out_csv = per_model_dir / f"{suffix}.csv"
        if out_csv.exists():
            print(f"{tag}: complete, skipping", flush=True)
            continue
        for other in MODEL_TAGS.values():
            if other != tag:
                unload(other)
        with open(args.rag_dir / f"rag_responses_{suffix}.json", encoding="utf-8") as f:
            records = json.load(f)[: args.n]
        llm = ChatOllama(model=tag, **GEN_OPTIONS)
        # Warm-up call so model load time is not counted in the measured queries
        llm.invoke(build_messages(records[0]["user_input"], records[0]["retrieved_contexts"]))
        print(f"Placement before {tag}:\n{placement()}", flush=True)

        rows = []
        for idx, rec in enumerate(records):
            messages = build_messages(rec["user_input"], rec["retrieved_contexts"])
            start = time.perf_counter()
            reply = llm.invoke(messages)
            wall = time.perf_counter() - start
            meta = reply.response_metadata
            eval_count = meta.get("eval_count")
            eval_s = ns_to_s(meta.get("eval_duration"))
            rows.append({
                "model": tag,
                "query_index": idx,
                "prompt_tokens": meta.get("prompt_eval_count"),
                "output_tokens": eval_count,
                "wall_s": round(wall, 3),
                "total_s": ns_to_s(meta.get("total_duration")),
                "eval_s": eval_s,
                "output_tokens_per_s": round(eval_count / eval_s, 2) if eval_count and eval_s else None,
            })
            print(f"{tag} [{idx + 1}/{len(records)}] wall={wall:.2f}s out={eval_count}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        unload(tag)

    frames = [pd.read_csv(p) for p in sorted(per_model_dir.glob("*.csv"))]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.output_dir / "per_query_timing.csv", index=False)

    summary = []
    for tag, group in frame.groupby("model", sort=False):
        summary.append({
            "model": tag,
            "n": len(group),
            "prompt_tokens_median": group["prompt_tokens"].median(),
            "output_tokens_median": group["output_tokens"].median(),
            "output_tokens_total": int(group["output_tokens"].sum()),
            "wall_s_median": round(group["wall_s"].median(), 3),
            "wall_s_p90": round(group["wall_s"].quantile(0.9), 3),
            "output_tokens_per_s_median": round(group["output_tokens_per_s"].median(), 2),
        })
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(args.output_dir / "summary_timing.csv", index=False)
    print(summary_frame.to_string(index=False))
    print(f"Saved outputs under {args.output_dir}.")


if __name__ == "__main__":
    main()
