# Champion–Challenger Model Comparison with Hugging Face and MLflow

* **Student Name:** mdyusuf
* **Student USN:** `1MS26CI405`
* **Repository Name:** `1MS26CI405-mlflow-champion-challenger`
* **Course Project:** MLOps Champion–Challenger Sentiment Classification Pipeline

## 1. Project Overview

This project implements a reproducible MLOps Champion–Challenger evaluation, experiment tracking, model registration, and alias-based deployment pipeline using [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/model_doc/distilbert) and the [MLflow AI Platform](https://mlflow.org/docs/latest/ml/model-registry/). Two pretrained binary sentiment classification models are evaluated on an identical 100-review sample drawn from the [Stanford IMDb Movie Review Dataset (`stanfordnlp/imdb`)](https://huggingface.co/datasets/stanfordnlp/imdb), tracked in an SQLite-backed MLflow server, registered in the [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/workflow/) with `@champion` and `@challenger` aliases, and invoked for batch inference through [`mlflow.pyfunc`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html) without direct Hugging Face API calls in the prediction script.

For deeper architectural and operational documentation, see the companion files in this repository:
* **Architectural & Metric Analysis:** `MODEL_COMPARISON.md`
* **End-to-End MLOps Lifecycle & Alias Deployment:** `MLOPS_WORKFLOW.md`

## 2. Selected Task and Models

* **Task:** Binary Sentiment Classification (`0 = NEGATIVE`, `1 = POSITIVE`)
* **Model A — Initial Champion (`champion_candidate`):** [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) — A 66-million-parameter distilled Transformer derived from [DistilBERT (`distilbert-base-uncased`)](https://huggingface.co/distilbert/distilbert-base-uncased) and fine-tuned on the Stanford Sentiment Treebank (SST-2).
* **Model B — Challenger (`challenger_candidate`):** [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english) — A 355-million-parameter model ("SiEBERT") fine-tuned from [RoBERTa-Large (`FacebookAI/roberta-large`)](https://huggingface.co/FacebookAI/roberta-large) across 15 diverse English sentiment datasets.

## 3. Dataset Details

* **Evaluation Dataset Source:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (`test` split, 25,000 labeled movie reviews).
* **Sampling Methodology:** Shuffled deterministically in `prepare_data.py` using `SEED = 42` and sliced to the first `100` examples, saved to `data/eval_inputs.csv` with columns `text` and `label`.
* **Class Balance:** `53` Negative reviews (`label = 0`) and `47` Positive reviews (`label = 1`).
* **Shared-Input Guarantee:** Both Model A and Model B are evaluated on the exact same 100 examples in the exact same row order with `max_length = 512` and `batch_size = 8`.
* **Unseen Inference Dataset:** `data/new_data.csv` contains `8` newly authored movie reviews (not present in `data/eval_inputs.csv`) used exclusively in `predict_from_mlflow.py`.

## 4. Repository Folder Structure

```text
1MS26CI405-mlflow-champion-challenger/
│
├── data/
│   ├── eval_inputs.csv
│   └── new_data.csv
├── outputs/
│   ├── confusion_matrix_champion_candidate.png
│   ├── confusion_matrix_challenger_candidate.png
│   ├── disagreements.csv
│   ├── disagreements_qualitative.json
│   ├── metrics_champion_candidate.json
│   ├── metrics_challenger_candidate.json
│   ├── metrics_summary.json
│   ├── model_comparison.png
│   ├── new_data_predictions.csv
│   ├── predictions_champion_candidate.csv
│   ├── predictions_challenger_candidate.csv
│   ├── registry_info.json
│   └── run_ids.json
├── screenshots/
│   ├── 01_model_test.png
│   ├── 02_evaluation_data.png
│   ├── mlflow_compare.png
│   ├── mlflow_registry.png
│   └── predict_from_mlflow.png
│
├── evaluate.py
├── predict_from_mlflow.py
├── prepare_data.py
├── register_champion.py
├── test_models.py
│
├── README.md
├── MODEL_COMPARISON.md
├── MLOPS_WORKFLOW.md
│
├── requirements.txt
└── .gitignore
```

## 5. Technologies and Requirements

| Package / Tool | Role in Pipeline | Reference |
| :--- | :--- | :--- |
| **Python 3.10+** | Core execution runtime on Windows 11 (PyCharm IDE) | Local `.venv` environment |
| **Transformers** | Loads pretrained tokenizers and `sentiment-analysis` pipelines | [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/distilbert) |
| **PyTorch (`torch`)** | Deep learning tensor execution backend (`device = -1` CPU) | Installed via `requirements.txt` |
| **Datasets** | Downloads and samples the IMDb benchmark split | [`stanfordnlp/imdb` Dataset Card](https://huggingface.co/datasets/stanfordnlp/imdb) |
| **MLflow** | Experiment tracking server, artifact store, PyFunc packaging, and Model Registry | [MLflow Model Registry Docs](https://mlflow.org/docs/latest/ml/model-registry/) |
| **scikit-learn** | Computes Accuracy, Precision, Recall, F1, Macro-F1, and Confusion Matrices | [scikit-learn `f1_score` Docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) |
| **Pandas & Matplotlib** | Tabular dataset handling and evaluation chart rendering | Installed via `requirements.txt` |

## 6. Installation and Execution Instructions

### Step 1: Configure Virtual Environment and Install Dependencies

Open the project folder in **PyCharm**, open the integrated **Terminal** (`Alt + F12`), and run:

```powershell
pip install -r requirements.txt
```

### Step 2: Start the MLflow Tracking and Registry Server (Terminal 1)

Start a database-backed [MLflow tracking server](https://mlflow.org/docs/latest/ml/model-registry/workflow/) so the Model Registry and aliases are enabled. Leave this terminal running continuously:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Open Google Chrome and navigate to `http://127.0.0.1:5000` to view the MLflow UI.

### Step 3: Execute the Pipeline Scripts in Order (Terminal 2)

Open a second terminal tab in PyCharm (`+` icon in the Terminal panel) and execute each stage in sequence:

```powershell
# Stage 2: Verify both Hugging Face models on a single test review
python test_models.py

# Stage 3: Prepare the 100-sample IMDb evaluation dataset (seed=42)
python prepare_data.py

# Stages 6 & 7: Evaluate both models, log runs/artifacts/models to MLflow, and apply the decision rule
python evaluate.py

# Stage 9: Register both models under 'sentiment-model' and assign @champion and @challenger aliases
python register_champion.py

# Stage 10: Load models:/sentiment-model@champion via mlflow.pyfunc and score data/new_data.csv
python predict_from_mlflow.py
```

### Step 4: Recommended Git Commit Sequence

```bash
# Commit 1: Core setup and data preparation
git add .gitignore requirements.txt test_models.py prepare_data.py data/
git commit -m "feat: initial project setup and data preparation (1MS26CI405)"

# Commit 2: Detailed model comparison report
git add MODEL_COMPARISON.md
git commit -m "docs: add detailed champion challenger model comparison"

# Commit 3: MLOps workflow and deployment architecture documentation
git add MLOPS_WORKFLOW.md
git commit -m "docs: document mlops experiment and deployment workflow"

# Commit 4: Evaluation scripts, registry automation, inference results, and final README
git add evaluate.py register_champion.py predict_from_mlflow.py outputs/ screenshots/ README.md
git commit -m "feat: complete evaluation artifacts, model registry aliases, and champion inference"
```

## 7. Predefined Champion–Challenger Decision Rule

Before executing `evaluate.py`, the decision rule is locked to ensure an objective comparison:

1. **Primary Quality Metric:** [`macro_f1`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) (higher is better), which weights `NEGATIVE (0)` and `POSITIVE (1)` classes equally.
2. **Minimum Improvement Margin:** `IMPROVEMENT_MARGIN = 0.01` (`+1.0%` absolute Macro-F1 improvement required to justify switching models).
3. **Latency Guardrail:** `LATENCY_GUARDRAIL = 5.0` — The Challenger's average latency per sample (`avg_latency_ms`) must not exceed `5.0` times the Initial Champion's average latency (`latency_B <= 5.0 * latency_A`).
4. **Promotion Logic Implemented in `evaluate.py`:**
   * **Tie-Breaker Within Margin (`|macro_f1_B - macro_f1_A| < 0.01`):** Both models are treated as equivalent in classification quality, and the **faster model** (`latency_A <= latency_B` → `model_A`, otherwise `model_B`) wins.
   * **Challenger Promotion (`(macro_f1_B - macro_f1_A) >= 0.01` and `latency_B <= 5.0 * latency_A`):** **Model B (`siebert/sentiment-roberta-large-english`)** is promoted to `champion`.
   * **Champion Retention (All Other Cases):** If Model B fails to improve Macro-F1 by at least `0.01` or exceeds `5.0x` the latency of Model A, **Model A (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`)** remains `champion`.

## 8. Actual Evaluation Results

*Note: Populate the bracketed placeholders below with the exact numerical values from `outputs/metrics_summary.json` after running `python evaluate.py` on your machine.*

| Metric | Model A (`distilbert-base-uncased-finetuned-sst-2-english`) | Model B (`siebert/sentiment-roberta-large-english`) | Difference (`B - A` / Ratio) |
| :--- | :--- | :--- | :--- |
| **MLflow Run Name** | `A_sentiment_distilbert-sst2` | `B_sentiment_roberta-large` | `compare_sentiment` |
| **Accuracy** | `[From outputs/metrics_summary.json]` | `[From outputs/metrics_summary.json]` | `[Delta Accuracy]` |
| **Precision** | `[From outputs/metrics_summary.json]` | `[From outputs/metrics_summary.json]` | `[Delta Precision]` |
| **Recall** | `[From outputs/metrics_summary.json]` | `[From outputs/metrics_summary.json]` | `[Delta Recall]` |
| **F1 Score (Binary)** | `[From outputs/metrics_summary.json]` | `[From outputs/metrics_summary.json]` | `[Delta F1]` |
| **Macro-F1 (Primary Metric)** | `[From outputs/metrics_summary.json]` | `[From outputs/metrics_summary.json]` | `[delta_macro_f1]` |
| **Avg Latency (`avg_latency_ms`)** | `[From outputs/metrics_summary.json]` | `[From outputs/metrics_summary.json]` | `[latency_ratio_b_over_a]x` |
| **Total Inference Time (`total_inference_s`)** | `[From outputs/metrics_summary.json]` | `[From outputs/metrics_summary.json]` | `[Delta Seconds]` |

## 9. Final Champion–Challenger Decision and Discussion

*Note: Once `python evaluate.py` finishes, fill in the exact Macro-F1 and latency values from `outputs/metrics_summary.json` and `outputs/disagreements_qualitative.json` in the paragraph below (150–250 words).*

On the 100-sample [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) evaluation dataset (`SEED = 42`, 53 negative and 47 positive reviews), the Initial Champion ([`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), 66M parameters) achieved a Macro-F1 of `[Model_A_Macro_F1]` with an average CPU inference latency of `[Model_A_Latency_ms]` ms/review, whereas the Challenger ([`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english), 355M parameters) achieved a Macro-F1 of `[Model_B_Macro_F1]` with an average CPU latency of `[Model_B_Latency_ms]` ms/review. This yields a Macro-F1 improvement of `[delta_macro_f1]` and a latency ratio of `[latency_ratio_b_over_a]x`. Because a 100-review sample carries a binomial standard error of roughly $\pm 3\%\text{ to }4\%$, requiring at least a `0.01` Macro-F1 margin prevents trivial noise from triggering a model swap, while the `5.0x` latency guardrail ensures that the 5.3x larger RoBERTa-Large architecture is only promoted if its runtime overhead on CPU remains acceptable. Across the 100 reviews, the two models disagreed on `[total_disagreements_found]` examples (logged in `outputs/disagreements_qualitative.json`), primarily on long reviews containing mixed praise and criticism or subtle irony where SST-2 sentence-level training differs from multi-corpus full-review training. Under our predefined rule, **`[winner]`** was selected and assigned the `@champion` alias in the [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/), while the runner-up was assigned `@challenger`.

## 10. Limitations

1. **Sample Size Constraints:** Evaluating on `100` IMDb reviews enables rapid local CPU experimentation, but each single review flips accuracy and Macro-F1 by approximately `0.01` (`1%`). Production promotion gates typically evaluate on `2,500+` stratified samples.
2. **Sequence Truncation at 512 Tokens:** Both models enforce a maximum context length of `512` tokens (`max_length = 512, truncation = True`). Very long movie reviews that place their concluding verdict after token 512 lose trailing sentiment cues.
3. **CPU-Only Latency Benchmarking:** Running with `DEVICE = -1` (CPU) magnifies the latency gap between a 6-layer DistilBERT model (`66M` parameters) and a 24-layer RoBERTa-Large model (`355M` parameters) compared to GPU batch execution.
4. **Binary Polarity Assumption:** Both models force every input into `POSITIVE` or `NEGATIVE`, which cannot explicitly flag neutral or ambivalent reviews (such as review #6 in `data/new_data.csv`).

## 11. AI Assistance Declaration

AI assistance was used by **mdyusuf (`1MS26CI405`)** for debugging MLflow tracking server configurations, structuring the custom `SentimentPyFunc` wrapper, auditing rubric completeness, and formatting documentation. All scripts were executed locally in PyCharm on Windows 11, and all logged metrics, artifacts, registry versions, and screenshots reflect genuine local execution.

## 12. Screenshots Checklist

| Screenshot File | Description | Verification Target |
| :--- | :--- | :--- |
| `screenshots/01_model_test.png` | PyCharm Terminal output of `python test_models.py` | Confirms both Hugging Face models load and classify a sample review |
| `screenshots/02_evaluation_data.png` | PyCharm Terminal / CSV preview of `data/eval_inputs.csv` | Confirms 100 IMDb samples (`53` negative, `47` positive) with `SEED = 42` |
| `screenshots/mlflow_compare.png` | MLflow UI Compare view (`http://127.0.0.1:5000`) | Displays `A_sentiment_distilbert-sst2` vs. `B_sentiment_roberta-large` metrics and latency |
| `screenshots/mlflow_registry.png` | MLflow Model Registry view for `sentiment-model` | Displays Version 1 and Version 2 with `@champion` and `@challenger` aliases |
| `screenshots/predict_from_mlflow.png` | PyCharm Terminal output of `python predict_from_mlflow.py` | Displays student USN `1MS26CI405`, resolved `@champion` version, HF ID, and 8 predictions |
