# %%
"""Generate RAG answers for the evaluation subset with fixed, recorded settings.

Retrieval is frozen: the retrieved contexts come from a saved response file and are
verified against the testset questions, so the generator is the only varying component.
Resumable: each model appends to a partial JSONL file and is skipped when complete.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from langchain_ollama import ChatOllama
import pandas as pd

MODEL_TAGS = {
    "gemma3_4b": "gemma3:4b",
    "glm-4.7-flash_q4_K_M": "glm-4.7-flash:q4_K_M",
    "ministral-3_8b": "ministral-3:8b",
    "nemotron-3-nano_4b": "nemotron-3-nano:4b",
    "qwen3_4b-instruct": "qwen3:4b-instruct",
}

SYSTEM = "You are a helpful assistant that answers questions based on given documents only."

# Decoding and context settings, identical for every model
GEN_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024,
               "reasoning": False}  # thinking disabled: hidden reasoning otherwise uses the token budget


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_messages(query, context):
    # Same prompt as generate_answer() in the original pipeline
    prompt = f"question: {query}\n\nDocuments: {context}"
    return [("system", SYSTEM), ("human", prompt)]


def load_frozen_items(testset_path, frozen_path, n):
    testset = pd.read_csv(testset_path, index_col=False).head(n)
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))[:n]
    items = []
    for (_, row), saved in zip(testset.iterrows(), frozen):
        if row["user_input"] != saved["user_input"]:
            sys.exit(f"Testset and frozen retrieval disagree on: {row['user_input'][:60]!r}")
        items.append({
            "user_input": row["user_input"],
            "reference": row["reference"],
            "retrieved_contexts": saved["retrieved_contexts"],
        })
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, default=Path("outputs/testset/01_esg_testset.csv"))
    parser.add_argument("--frozen-retrieval", type=Path,
                        default=Path("outputs/rag_responses/rag_responses_qwen3_4b-instruct.json"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/repro/rag_responses"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = load_frozen_items(args.testset, args.frozen_retrieval, args.n)
    manifest = {
        "testset": str(args.testset), "testset_sha256": sha256(args.testset),
        "frozen_retrieval": str(args.frozen_retrieval), "frozen_retrieval_sha256": sha256(args.frozen_retrieval),
        "n": args.n, "generation_options": GEN_OPTIONS, "system_prompt": SYSTEM,
        "models": MODEL_TAGS,
    }
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
        final = args.output_dir / f"rag_responses_{suffix}.json"
        partial = args.output_dir / f"rag_responses_{suffix}.partial.jsonl"
        if final.exists():
            print(f"{tag}: complete, skipping", flush=True)
            continue
        done = set()
        if partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                done.add(json.loads(line)["index"])
        llm = ChatOllama(model=tag, **GEN_OPTIONS)
        with open(partial, "a", encoding="utf-8") as out:
            for idx, item in enumerate(items):
                if idx in done:
                    continue
                reply = llm.invoke(build_messages(item["user_input"], item["retrieved_contexts"]))
                rec = {"index": idx, **item, "response": reply.content}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{tag} [{idx + 1}/{len(items)}]", flush=True)
        rows = []
        for line in partial.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            rows.append({k: rec[k] for k in ("user_input", "reference", "retrieved_contexts", "response")})
        final.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{tag}: saved {len(rows)} responses to {final}", flush=True)


if __name__ == "__main__":
    main()
