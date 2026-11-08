# ESG-RAG benchmark (RAGAS)

Code, data, and results for **"Empirical Evaluation of Open-Source Large Language Models for Retrieval-Augmented Generation in ESG Domain"** (IEEE BigData 2026, paper BigD275).

The repository evaluates five open-source generators in one shared RAG pipeline over 498 ESG reports. It scores each answer with seven RAGAS metrics and five LLM judges, under fixed decoding settings.

Public repository: https://github.com/datalab-unisalento/ESG-RAG-benchmark-ragas

## What is here

| Path | Contents |
|---|---|
| `data/documents/` | 498 ESG reports, converted to Markdown |
| `outputs/testset/` | Synthetic QA set: 284 pairs from 181 reports (`01_esg_testset.csv`, `.json`) |
| `outputs/repro2/` | **Controlled rerun** (the results in the paper): responses, judge scores, timing, environment record, logs |
| `outputs/repro/` | Earlier run with thinking enabled (GLM answers were often empty). Superseded; kept for the audit trail |
| `outputs/evaluation/`, `outputs/100/`, `outputs/judge_sensitivity*/` | Earlier runs, kept for the audit trail |
| `outputs/qa_validation*/` | QA consistency checks (282-pair run and 284-pair run) |
| `outputs/benchmark*/` | Token and latency measurements |
| `code/` | All scripts; see below |
| `code/run_reproduction.sh` | Runs the controlled rerun stage by stage, resumable |

## Models used

**1. Data creation (synthetic QA pairs)**
- `qwen3:4b-instruct` wrote the questions and reference answers.
- `qwen3-embedding:4b` built the embedding index for the knowledge graph.

**2. Answer generation (models benchmarked)**
- glm-4.7-flash (`glm-4.7-flash:q4_K_M`, 29.9B mixture-of-experts, 4-bit)
- ministral-3 (`ministral-3:8b`)
- gemma3 (`gemma3:4b`)
- qwen3 (`qwen3:4b-instruct`)
- nemotron-3-nano (`nemotron-3-nano:4b`)

**3. Evaluation (RAGAS scoring of the answers)**
- Main judge: `qwen3:4b-instruct`. It is also the data-creation model, a self-preference risk the paper discusses.
- Independent judges: `nemotron-3.5-lightning:30b`, `cogito:32b`, `granite4.2:30b`, and `lfm2.5:24b`.
- Embeddings: `qwen3-embedding:4b`, for retrieval and answer relevancy.

**Also used in evaluation (QA consistency check on the reference answers):** `gemma4:12b`, `ministral-3:8b`, and `nemotron-3.5-lightning:30b`.

## Controlled settings (used for all reported results)

- **Generation:** temperature 0, seed 42, 8,192-token context, 1,024 output tokens, for every model.
- **Retrieval:** top-one chunk by cosine similarity, frozen from the original retrieval run, identical for all models and checked against the testset questions.
- **Judging:** RAGAS 0.4.3 with five judges (qwen3:4b-instruct as main judge; nemotron-3.5-lightning:30b, cogito:32b, granite4.2:30b, and lfm2.5:24b as independent judges), temperature 0, seed 42, 8,192-token context, thinking disabled.
- **Models:** glm-4.7-flash (q4_K_M, 29.9B mixture-of-experts), ministral-3:8b, gemma3:4b, qwen3:4b-instruct, nemotron-3-nano:4b.

The exact package versions, GPU, and Ollama model list are recorded in `outputs/repro2/environment.json`, `outputs/repro2/pip_freeze.txt`, and `outputs/repro2/ollama_models.txt`. `requirements.txt` lists the packages without pins; use `pip_freeze.txt` for the exact versions.

## Reproducing the results

Requirements: Python 3.12, [Ollama](https://ollama.com) running locally, and a GPU with about 48 GB of memory.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull qwen3:4b-instruct qwen3-embedding:4b gemma3:4b glm-4.7-flash:q4_K_M \
            nemotron-3-nano:4b ministral-3:8b nemotron-3.5-lightning:30b cogito:32b granite4.2:30b lfm2.5:24b gemma4:12b
bash code/run_reproduction.sh
```

`run_reproduction.sh` runs five stages: environment record, RAG generation, judging (all seven metrics, three judges in this script), token and latency timing, and the QA consistency check. Each stage resumes from its own manifest, so an interrupted run continues where it stopped. Logs are written to `outputs/repro/logs/`.

Run it detached if you need to disconnect:

```bash
setsid nohup bash code/run_reproduction.sh > outputs/repro/driver.log 2>&1 < /dev/null &
```

Stage durations depend on the hardware. The logs in `outputs/repro2/logs/` and the timing in `outputs/repro2/benchmark/` record the durations on our workstation.

**Note on the reported run.** The paper's numbers come from `outputs/repro2/`. Judging there used all five judges with thinking disabled; the judge list is recorded in `outputs/repro2/judges_nothink/manifest.json`. The judging command was:

```bash
.venv/bin/python code/04_compare_ragas_judges.py --input-dir outputs/repro2/rag_responses \
    --output-dir outputs/repro2/judges_nothink \
    --judges qwen3:4b-instruct nemotron-3.5-lightning:30b cogito:32b granite4.2:30b lfm2.5:24b
```

`run_reproduction.sh` predates the final run: it writes to `outputs/repro/` and judges with three models only.

## Scripts

| Script | Purpose |
|---|---|
| `code/02_run_rag_generation.py` | Generation with frozen retrieval and fixed decoding (resumable) |
| `code/04_compare_ragas_judges.py` | RAGAS scoring by one or more judges, all seven metrics (resumable) |
| `code/05_validate_synthetic_testset.py` | Consistency check of the synthetic QA references (resumable) |
| `code/06_benchmark_tokens_latency.py` | Token and latency measurement, one model loaded at a time (resumable) |
| `code/07_ragas_version_check.py` | Check that the installed RAGAS version reproduces saved scores |
| `code/08_summarize_repro.py` | Judge agreement and paired bootstrap intervals (seed 42) |
| `code/09_judge_repeatability.py` | Compare two scorings of the same responses by the same judge |
| `code/00_select_documents.py`, `code/01_generate_testset.py` | Document selection and synthetic QA generation (historical; see notes) |

## Metrics

Seven RAGAS metrics: context precision (with and without reference), context recall, context relevance, faithfulness, answer relevancy, and factual correctness (F1). Formulas are in the Evaluation Metrics subsection of the paper.

## Main findings

- Faithfulness: ministral-3 is the least faithful model under all five judges. glm-4.7-flash and gemma3 are the most faithful, and their order depends on the judge.
- Answer relevancy: glm-4.7-flash has the lowest score under three of the five judges, and the differences are small and not significant.
- Factual correctness: no model ranks first under all five judges. nemotron-3-nano ranks first under the main, cogito, and lfm2.5 judges. The margins are small and not significant after correction. The earlier qwen3 lead does not reproduce under the controlled settings.
- Retrieval scores depend on the judge, and context recall is undefined for most items, so the paper draws no retrieval ranking.
- Repeatability: rescoring with the same settings reproduced 87% of judge-based scores exactly; model means moved by at most 0.03.

## Notes and limitations

- Synthetic QA references are checked only by LLM judges. No human verified them.
- The main judge (qwen3) also generated the questions. Four further judges test this risk, but cannot fully separate judge capability from self-preference. cogito uses the Qwen2 architecture, and nemotron-3.5 is from the same vendor as nemotron-3-nano.
- The evaluation uses a 100-item prefix, not a random sample.
- The retrieval index (embeddings for all 258,590 chunks) is not included, so the retrieval step cannot be rebuilt from scratch. The frozen retrieved contexts are included in `outputs/rag_responses/` and `outputs/repro2/rag_responses/`.
- The earlier runs in `outputs/evaluation/` and `outputs/100/` used the models' default decoding and an earlier RAGAS setup. They are kept for the audit trail and are not the reported results.
- The synthetic QA set was regenerated once. `outputs/testset/01_esg_testset.csv` is the 284-pair version used by the paper. A 282-pair version from the later regeneration is in the git history.

## Citation

If you use the data, code, or results, please cite the paper (IEEE BigData 2026, paper BigD275). A BibTeX entry will be added after publication.

## Licence

- **Code** (`code/`, `README.md`, and the other scripts): MIT License, see `LICENSE`.
- **Generated data and results** (`outputs/`, including the synthetic QA pairs, RAG responses, judge scores, timing, and logs): Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- **ESG reports** (`data/documents/`): copies of publicly available company reports. Their rights remain with the respective publishers. They are provided for research reproduction only; contact the authors to request removal of any document.
