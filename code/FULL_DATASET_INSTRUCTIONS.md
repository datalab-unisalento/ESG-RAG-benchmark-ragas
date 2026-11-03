# Full Dataset Evaluation Instructions

## Overview
Code has been modified to run evaluation on the full dataset (498 documents) instead of the 100-document sample.

## Changes Made

### 1. Document Selection Script (`00_select_documents.py`)

**Modified Functions:**
- **Removed**: `stratified_sample_documents()` function
- **Added**: `select_all_documents()` function that returns all documents sorted by year and size
- **Updated**: `main()` function to use full dataset selection
- **Changed**: Output directory from `markdown_docs_100` to `markdown_docs_full`

**Key Changes:**
```python
def select_all_documents(df):
    """Select all documents for full dataset evaluation"""
    # Return all documents sorted by year and size for consistency
    selected_docs = df.sort_values(['year', 'size']).to_dict('records')
    return selected_docs
```

### 2. Testset Generation Script (`01_generate_testset.py`)

**Modified:**
- **Changed**: Path from `markdown_docs_100` to `markdown_docs_full`
- **Updated**: Comment to reflect full dataset usage

**Key Change:**
```python
# Use full dataset for experiments
path = "markdown_docs_full"
```

## Running the Full Dataset Evaluation

### Step 1: Generate Full Document Dataset
```bash
cd code
python 00_select_documents.py
```

**Expected Output:**
- Creates `markdown_docs_full/` directory with all 498 documents
- Generates `outputs/document_selection_info.csv` with full dataset info
- Generates `outputs/selected_documents.json` with all document metadata

### Step 2: Generate Testset on Full Dataset
```bash
python 01_generate_testset.py
```

**Expected Output:**
- Creates `outputs/01_esg_testset/01_esg_testset.csv` with evaluation queries
- Creates `outputs/01_esg_testset/01_esg_knowledge_graph.pkl` with document embeddings

### Step 3: Run RAG Evaluation
```bash
python 02_run_rag_evaluation.py
```

**Expected Output:**
- Evaluates all 5 models on full 498-document dataset
- Creates detailed results in `outputs/02_rag_responses/`
- Generates evaluation metrics for each model

### Step 4: Analyze Results
```bash
python 03_analyze_results.py
```

**Expected Output:**
- Creates summary results in `outputs/03_evaluation_results/`
- Generates performance comparison tables and analysis

## Benefits of Full Dataset Evaluation

### Addressing Reviewer Critiques:
1. **Sample Size**: Increases from 100 to 498 documents (5x increase)
2. **Statistical Power**: More robust statistical significance
3. **Generalizability**: Better representation across years and document types
4. **Reliability**: More stable performance estimates

### Expected Impact on Results:
- **Performance Metrics**: More stable and reliable estimates
- **Confidence Intervals**: Narrower due to larger sample size
- **Statistical Tests**: Higher power to detect significant differences
- **Model Comparison**: More robust ranking of model performance

## Computational Considerations

### Processing Time:
- **Document Selection**: ~2-3 minutes (vs ~30 seconds for 100 docs)
- **Testset Generation**: ~30-45 minutes (vs ~10-15 minutes for 100 docs)
- **RAG Evaluation**: ~4-6 hours (vs ~1-1.5 hours for 100 docs)
- **Total**: ~5-7 hours (vs ~1.5-2 hours for 100 docs)

### Resource Requirements:
- **Memory**: Higher RAM usage for larger embeddings
- **Storage**: Larger intermediate files
- **GPU**: Longer GPU utilization periods

## Quality Assurance

### Validation Steps:
1. **Document Count**: Verify 498 documents in `markdown_docs_full/`
2. **Testset Size**: Confirm proportional increase in evaluation queries
3. **Results Completeness**: Ensure all models evaluated on full dataset
4. **Performance Comparison**: Compare with 100-document baseline

### Expected Improvements:
- **More Reliable Results**: Reduced variance in performance metrics
- **Better Statistical Significance**: Ability to detect smaller effects
- **Stronger Paper**: Addresses major reviewer concern about sample size
- **Publication Readiness**: More robust experimental validation

## Notes
- The evaluation framework remains identical - only dataset size changes
- All 5 models will be evaluated on the same 498 documents
- Statistical analysis will have much higher power
- Results will be more suitable for academic publication
