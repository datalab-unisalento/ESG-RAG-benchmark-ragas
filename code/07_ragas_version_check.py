# %%
"""Check that the installed RAGAS version reproduces saved scores.

Rescores the first N saved responses of one model with the same judge, embeddings,
and metrics as 03_analyze_results.py, then compares each metric with the saved
per-item scores from the original run. Run inside the project virtualenv.
"""
import argparse
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

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

# Saved per-item score keys -> RAGAS metric class (same set as 03_analyze_results.py)
METRIC_KEYS = {
    "llm_context_precision_with_reference": LLMContextPrecisionWithReference,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "nv_context_relevance": ContextRelevance,
    "faithfulness": Faithfulness,
    "factual_correctness(mode=f1)": FactualCorrectness,
    "answer_relevancy": AnswerRelevancy,
}


def is_number(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-file", type=Path, default=Path("outputs/rag_responses/rag_responses_qwen3_4b-instruct.json"))
    parser.add_argument("--saved", type=Path, default=Path("outputs/evaluation/evaluation_results_qwen3_4b-instruct_20260512_172800.json"))
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/ragas_version_check/result.json"))
    args = parser.parse_args()

    responses = json.load(open(args.rag_file))[: args.n]
    saved = json.load(open(args.saved))[: args.n]

    dataset = EvaluationDataset.from_list(responses)
    llm = LangchainLLMWrapper(ChatOllama(model="qwen3:4b-instruct"))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="qwen3-embedding:4b"))
    metrics = [cls() for cls in METRIC_KEYS.values()]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=3600, max_workers=1),
    )
    new_rows = result.scores

    report = {"ragas_version": version("ragas"), "n_rows": len(responses), "metrics": {}}
    for key in METRIC_KEYS:
        old = [row.get(key) for row in saved]
        new = [row.get(key) for row in new_rows]
        pairs = [(o, n) for o, n in zip(old, new) if is_number(o) and is_number(n)]
        diffs = [abs(o - n) for o, n in pairs]
        report["metrics"][key] = {
            "saved_mean": mean([v for v in old if is_number(v)]),
            "rescored_mean": mean([v for v in new if is_number(v)]),
            "paired_items": len(pairs),
            "max_abs_diff": max(diffs) if diffs else None,
            "mean_abs_diff": mean(diffs),
            "saved_per_item": old,
            "rescored_per_item": new,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"RAGAS {report['ragas_version']}, rows {report['n_rows']}")
    for key, m in report["metrics"].items():
        print(f"{key:40s} saved={m['saved_mean']} rescored={m['rescored_mean']} "
              f"paired={m['paired_items']} max_diff={m['max_abs_diff']}")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
