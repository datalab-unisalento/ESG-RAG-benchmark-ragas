#!/usr/bin/env python3
"""Re-score saved RAG responses with independent local Ollama judges."""

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path

try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    import langchain_google_vertexai

    sys.modules["langchain_community.chat_models.vertexai"] = (
        langchain_google_vertexai
    )

import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
    FactualCorrectness,
    LLMContextPrecisionWithReference,
)


DEFAULT_JUDGES = ["nemotron-3.5-lightning:30b", "cogito:32b"]
# Fixed judge settings, recorded in the run manifest
JUDGE_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}
METRICS = {
    "llm_context_precision_with_reference": (
        LLMContextPrecisionWithReference,
        "llm_context_precision_with_reference",
    ),
    "context_precision": (ContextPrecision, "context_precision"),
    "context_recall": (ContextRecall, "context_recall"),
    "context_relevance": (ContextRelevance, "nv_context_relevance"),
    "faithfulness": (Faithfulness, "faithfulness"),
    "factual_correctness": (FactualCorrectness, "factual_correctness(mode=f1)"),
    "answer_relevancy": (AnswerRelevancy, "answer_relevancy"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two or more independent judges on saved RAG outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/rag_responses"),
        help="Directory containing rag_responses_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/judge_sensitivity"),
        help="Directory for checkpointed scores and summaries.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
        help="RAGAS metrics to run (default: all seven).",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=DEFAULT_JUDGES,
        help="Installed Ollama judge model tags.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Rows per checkpointed evaluation batch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from chunk files after verifying the run manifest.",
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")
    if not args.judges or len(set(args.judges)) != len(args.judges):
        parser.error("--judges must contain at least one model tag, with no duplicates.")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalize_scores(scores, metric_names):
    normalized = []
    for row in scores:
        item = {}
        for metric in metric_names:
            value = row.get(metric)
            if pd.isna(value):
                item[metric] = None
            else:
                item[metric] = float(value)
        normalized.append(item)
    return normalized


def repo_relative(path):
    """Store paths relative to the working directory so manifests are portable."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def create_run_config(args, response_files):
    return {
        "judge_models": args.judges,
        "chunk_size": args.chunk_size,
        "response_files": [
            {
                "path": repo_relative(path),
                "sha256": sha256_file(path),
                "n_rows": len(json.loads(path.read_text(encoding="utf-8"))),
            }
            for path in response_files
        ],
        "embedding_model": "qwen3-embedding:4b",
        "metrics": args.metrics,
        "judge_options": JUDGE_OPTIONS,
    }


def chunk_path(output_dir, judge, rag_model, chunk_index):
    return output_dir / slug(judge) / slug(rag_model) / f"chunk_{chunk_index:04d}.json"


def load_chunk(path, expected_count):
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload["scores"]
    if len(scores) != expected_count:
        raise ValueError(f"{path} has {len(scores)} scores; expected {expected_count}.")
    return scores


def evaluate_chunk(rows, judge, embeddings, metric_names):
    dataset = EvaluationDataset.from_list(rows)
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=judge, **JUDGE_OPTIONS))
    metrics = [METRICS[name][0]() for name in metric_names]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=3600, max_workers=1),
    )
    score_names = [METRICS[name][1] for name in metric_names]
    normalized = normalize_scores(result.scores, score_names)
    return [
        {
            metric_name: row[score_name]
            for metric_name, score_name in zip(metric_names, score_names)
        }
        for row in normalized
    ]


def average_scores(rows, metric_names):
    averages = {}
    for metric in metric_names:
        values = [
            row[metric]
            for row in rows
            if row.get(metric) is not None and math.isfinite(row[metric])
        ]
        averages[metric] = sum(values) / len(values) if values else None
    return averages


def main():
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    response_files = sorted(args.input_dir.glob("rag_responses_*.json"))
    if not response_files:
        raise FileNotFoundError(f"No rag_responses_*.json files in {args.input_dir}.")

    manifest_path = args.output_dir / "manifest.json"
    manifest = create_run_config(args, response_files)
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(
                f"{manifest_path} does not match the requested run. "
                "Use a new --output-dir for different settings."
            )
        if not args.resume:
            raise FileExistsError(
                f"{args.output_dir} already contains a run; pass --resume."
            )
    else:
        if args.resume:
            raise ValueError("--resume requested but no matching manifest exists.")
        write_json(manifest_path, manifest)

    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=manifest["embedding_model"])
    )
    summaries = {}
    for judge in args.judges:
        summaries[judge] = []
        print(f"\n=== Judge: {judge} ===", flush=True)
        for response_file in response_files:
            rag_model = response_file.name.removeprefix("rag_responses_").removesuffix(
                ".json"
            )
            rows = json.loads(response_file.read_text(encoding="utf-8"))
            model_scores = []
            chunk_count = (len(rows) + args.chunk_size - 1) // args.chunk_size
            for chunk_index in range(chunk_count):
                start = chunk_index * args.chunk_size
                chunk_rows = rows[start : start + args.chunk_size]
                path = chunk_path(args.output_dir, judge, rag_model, chunk_index)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if not args.resume:
                        raise FileExistsError(
                            f"{path} already exists; pass --resume to continue."
                        )
                    scores = load_chunk(path, len(chunk_rows))
                    print(
                        f"{rag_model}: chunk {chunk_index + 1}/{chunk_count} "
                        "already complete",
                        flush=True,
                    )
                else:
                    print(
                        f"{rag_model}: scoring rows {start + 1}-"
                        f"{start + len(chunk_rows)} of {len(rows)}",
                        flush=True,
                    )
                    scores = evaluate_chunk(
                        chunk_rows, judge, embeddings, args.metrics
                    )
                    write_json(
                        path,
                        {
                            "judge_model": judge,
                            "rag_model": rag_model,
                            "start_row": start + 1,
                            "end_row": start + len(chunk_rows),
                            "scores": scores,
                        },
                    )
                model_scores.extend(scores)
            summaries[judge].append(
                {"Model": rag_model, **average_scores(model_scores, args.metrics)}
            )
            write_json(
                args.output_dir / f"summary_{slug(judge)}.json",
                {
                    "judge_model": judge,
                    "n_rag_models": len(summaries[judge]),
                    "results": summaries[judge],
                },
            )

        frame = pd.DataFrame(summaries[judge])
        frame.to_csv(args.output_dir / f"summary_{slug(judge)}.csv", index=False)
        print(f"Saved summary for {judge}.", flush=True)

    comparison = {}
    for judge in args.judges:
        comparison[judge] = {
            entry["Model"]: {
                metric: entry.get(metric) for metric in args.metrics
            }
            for entry in summaries[judge]
        }
    write_json(args.output_dir / "judge_comparison.json", comparison)
    print(f"\nAll judge scores saved under {args.output_dir}.", flush=True)


if __name__ == "__main__":
    main()
