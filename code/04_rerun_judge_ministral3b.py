import sys
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from ragas import RunConfig
import pandas as pd
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
import json
import os
import glob
from ragas.metrics import LLMContextPrecisionWithReference, ContextPrecision, ContextRecall, ContextRelevance,  Faithfulness, FactualCorrectness, AnswerRelevancy
from datetime import datetime


dir = "outputs/rag_responses"

model_files = glob.glob(f"{dir}/rag_responses_*.json")
models = []
for file_path in model_files:
    filename = os.path.basename(file_path)
    model_name = filename.replace("rag_responses_", "").replace(".json", "")
    models.append(model_name)

print(f"Found {len(models)} model files to process: {models}")

all_results = {}

for ragLLM in models:
    print(f"\n=== Evaluating {ragLLM} ===")

    with open(f"{dir}/rag_responses_{ragLLM}.json", "r") as f:
        ragas_dataset = json.load(f)

    evaluation_dataset = EvaluationDataset.from_list(ragas_dataset)

    llm = ChatOllama(model="ministral-3:3b", temperature=0)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    evaluator_llm = LangchainLLMWrapper(llm)
    my_metrics = [LLMContextPrecisionWithReference(), ContextPrecision(), ContextRecall(), ContextRelevance(), Faithfulness(), FactualCorrectness(), AnswerRelevancy()]

    result = evaluate(dataset=evaluation_dataset, metrics=my_metrics, llm=evaluator_llm, embeddings=generator_embeddings, run_config=RunConfig(timeout=3600, max_workers=1) )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"outputs/evaluation/evaluation_results_{ragLLM}_ministral3bjudge_{timestamp}.json", "w") as f:
        json.dump(result.scores, f)

    if hasattr(result, 'scores') and isinstance(result.scores, list):
        avg_scores = {}
        for score_dict in result.scores:
            for key, value in score_dict.items():
                if key not in avg_scores:
                    avg_scores[key] = []
                if pd.notna(value):
                    avg_scores[key].append(float(value))

        final_scores = {}
        for key, values in avg_scores.items():
            final_scores[key] = sum(values) / len(values) if values else 0.0

        all_results[ragLLM] = final_scores
    elif hasattr(result, 'scores'):
        all_results[ragLLM] = result.scores
    else:
        all_results[ragLLM] = {}

print("\n=== SUMMARY RESULTS (ministral-3:3b judge) ===")

summary_data = []
for model, scores in all_results.items():
    summary_data.append({
        'Model': model,
        'LLMContextPrecisionWithReference': scores.get('llm_context_precision_with_reference', 0),
        'ContextPrecision': scores.get('context_precision', 0),
        'ContextRecall': scores.get('context_recall', 0),
        'ContextRelevance': scores.get('nv_context_relevance', 0),
        'Faithfulness': scores.get('faithfulness', 0),
        'FactualCorrectness': scores.get('factual_correctness(mode=f1)', 0),
        'AnswerRelevancy': scores.get('answer_relevancy', 0)
    })

summary_df = pd.DataFrame(summary_data)

print("\nModel Performance Summary (ministral-3:3b judge):")
print(summary_df.to_string(index=False, float_format='%.3f'))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

summary_csv_path = f"outputs/evaluation/summary_results_ministral3bjudge_{timestamp}.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSummary results saved to: {summary_csv_path}")

summary_json = {
    'summary_table': summary_data,
    'timestamp': timestamp,
    'judge_model': 'ministral-3:3b'
}

summary_json_path = f"outputs/evaluation/detailed_summary_ministral3bjudge_{timestamp}.json"
with open(summary_json_path, "w") as f:
    json.dump(summary_json, f, indent=2)

print(f"Detailed summary saved to: {summary_json_path}")
