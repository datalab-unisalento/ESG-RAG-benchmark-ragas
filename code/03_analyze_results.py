# %%
import sys
# Compatibility: newer langchain-community moved ChatVertexAI to langchain-google-vertexai
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
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, ContextPrecision, ContextRecall, ContextRelevance,  Faithfulness, FactualCorrectness, AnswerRelevancy
from datetime import datetime


dir = "outputs/rag_responses"

# Automatically discover model files in directory
model_files = glob.glob(f"{dir}/rag_responses_*.json")
models = []
for file_path in model_files:
    # Extract model name from filename: rag_responses_model_name.json -> model_name
    filename = os.path.basename(file_path)
    model_name = filename.replace("rag_responses_", "").replace(".json", "")
    models.append(model_name)

print(f"Found {len(models)} model files to process: {models}")

# Store all results for summary
all_results = {}

for ragLLM in models:
    print(ragLLM)

    with open(f"{dir}/rag_responses_{ragLLM}.json", "r") as f:
        ragas_dataset = json.load(f)

    evaluation_dataset = EvaluationDataset.from_list(ragas_dataset)


    llm = ChatOllama(model="qwen3:4b-instruct")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

    generator_embeddings = LangchainEmbeddingsWrapper(embeddings) 


    evaluator_llm = LangchainLLMWrapper(llm)
    my_metrics = [LLMContextPrecisionWithReference(), ContextPrecision(), ContextRecall(), ContextRelevance(), Faithfulness(), FactualCorrectness(), AnswerRelevancy()]

    result = evaluate(dataset=evaluation_dataset, metrics=my_metrics, llm=evaluator_llm, embeddings=generator_embeddings, run_config=RunConfig(timeout=3600, max_workers=1) )

    # Debug: print result structure
    print(f"Result type: {type(result)}")
    print(f"Result.scores type: {type(result.scores)}")
    if hasattr(result, 'scores'):
        print(f"Result.scores: {result.scores}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"outputs/evaluation/evaluation_results_{ragLLM}_{timestamp}.json", "w") as f:
        json.dump(result.scores, f)
    
    # Store results for summary - calculate averages from list of dictionaries
    if hasattr(result, 'scores') and isinstance(result.scores, list):
        # Calculate average scores from list of dictionaries
        avg_scores = {}
        for score_dict in result.scores:
            for key, value in score_dict.items():
                if key not in avg_scores:
                    avg_scores[key] = []
                if pd.notna(value):
                    avg_scores[key].append(float(value))
        
        # Calculate final averages
        final_scores = {}
        for key, values in avg_scores.items():
            final_scores[key] = sum(values) / len(values) if values else 0.0
        
        all_results[ragLLM] = final_scores
    elif hasattr(result, 'scores'):
        all_results[ragLLM] = result.scores
    else:
        all_results[ragLLM] = {}

# Calculate and save summary results
print("\n=== SUMMARY RESULTS ===")

# Create summary DataFrame
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

# Print summary table
print("\nModel Performance Summary:")
print(summary_df.to_string(index=False, float_format='%.3f'))

# Generate timestamp for summary files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save summary to CSV with timestamp
summary_csv_path = f"outputs/evaluation/summary_results_{timestamp}.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSummary results saved to: {summary_csv_path}")

# Save detailed summary to JSON with timestamp
summary_json = {
    'summary_table': summary_data,
    'timestamp': timestamp
}

summary_json_path = f"outputs/evaluation/detailed_summary_{timestamp}.json"
with open(summary_json_path, "w") as f:
    json.dump(summary_json, f, indent=2)

print(f"Detailed summary saved to: {summary_json_path}") 




