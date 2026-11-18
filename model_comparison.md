# Champion vs. Challenger Model Comparison Report

* **Student Name:** mdyusuf
* **Student USN:** `1MS26CI405`
* **Project:** Champion–Challenger Model Comparison with Hugging Face and MLflow
* **Experiment Name:** `hf-sentiment-champion-challenger`

## 1. Executive Summary

This report provides a comprehensive architectural, parametric, and operational comparison between our two candidate models for binary sentiment classification (`POSITIVE` vs. `NEGATIVE`) evaluated on the [Stanford IMDb Movie Review Dataset (`stanfordnlp/imdb`)](https://huggingface.co/datasets/stanfordnlp/imdb):

* **Model A — Initial Champion (`champion_candidate`):** [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
* **Model B — Challenger (`challenger_candidate`):** [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english)

In an MLOps production setting, selecting a model is rarely a one-dimensional search for the highest accuracy. Engineering teams must balance **predictive quality** against **computational cost, memory footprint, and inference latency**. This report explains the structural differences between DistilBERT (`66M` parameters) and RoBERTa-Large (`355M` parameters), evaluates their trade-offs and ideal deployment use cases, and justifies why **Macro-F1** ([`sklearn.metrics.f1_score(average="macro")`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)) serves as our primary decision metric.

## 2. DistilBERT vs. RoBERTa Architecture Comparison

Both candidate models belong to the encoder-only Transformer family introduced by BERT, meaning they process an entire input review bidirectionally using multi-head self-attention before passing the representation of the classification token (`[CLS]` in DistilBERT, `<s>` in RoBERTa) to a linear classification head. However, their pretraining paradigms and structural depth differ fundamentally.

### 2.1 Model A Architecture: DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)

[`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) is built on [DistilBERT (`distilbert-base-uncased`)](https://huggingface.co/distilbert/distilbert-base-uncased), which applies **knowledge distillation** during pretraining to compress BERT-base (`110M` parameters, 12 layers) into a compact student network (`66M` parameters, 6 layers) while retaining approximately 97% of BERT's downstream language understanding performance ([Hugging Face DistilBERT Documentation](https://huggingface.co/docs/transformers/en/model_doc/distilbert)):

1. **Knowledge Distillation Triple Loss:** The student network is trained to mimic the full output probability distribution (soft targets) and internal hidden states of the teacher network using a linear combination of three losses:
   * **Distillation Loss ($L_{\text{ce}}$):** Kullback–Leibler divergence / cross-entropy over the teacher's softened logits at temperature $T$.
   * **Masked Language Modeling Loss ($L_{\text{mlm}}$):** Standard cross-entropy predicting masked tokens.
   * **Cosine-Distance Loss ($L_{\text{cos}}$):** Aligns the direction of the student's and teacher's hidden state vectors across corresponding layers.
2. **Structural Simplifications:** According to the [DistilBERT Architecture Reference](https://huggingface.co/docs/transformers/en/model_doc/distilbert), DistilBERT halves the number of Transformer encoder layers (taking every second layer from the 12-layer teacher to form **6 layers**), removes token-type embeddings (`token_type_ids`), and eliminates the pooler layer.
3. **Downstream Fine-Tuning (SST-2):** The checkpoint [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) is fine-tuned exclusively on the **Stanford Sentiment Treebank (SST-2)**, a benchmark composed primarily of short, single-sentence movie review snippets.

### 2.2 Model B Architecture: SiEBERT / RoBERTa-Large (`siebert/sentiment-roberta-large-english`)

[`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english) ("SiEBERT" — *Sentiment in English*) is a fine-tuned checkpoint of [RoBERTa-Large (`FacebookAI/roberta-large`)](https://huggingface.co/FacebookAI/roberta-large), an optimized, scaled-up bidirectional Transformer architecture:

1. **RoBERTa Pretraining Optimizations:** Unlike original BERT, [RoBERTa-Large](https://huggingface.co/FacebookAI/roberta-large) removes the Next Sentence Prediction (NSP) objective, trains on an order-of-magnitude larger corpus (`160 GB` of uncompressed text) with larger mini-batches (`8,000` sequences), and uses **dynamic masking** (regenerating the masked token pattern every epoch rather than freezing static masks during data preprocessing).
2. **Byte-Level BPE Tokenization:** While [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) uses a `30,522`-token WordPiece vocabulary (lowercased/uncased), [RoBERTa-Large](https://huggingface.co/FacebookAI/roberta-large) uses a case-sensitive Byte-Level Byte-Pair Encoding (BPE) vocabulary of `50,265` subword tokens, allowing it to preserve capitalization emphasis (e.g., `"AMAZING"` vs. `"amazing"`) and represent rare words without falling back to `[UNK]` tokens.
3. **Deep 24-Layer Representation & Multi-Domain Fine-Tuning:** As documented on the [SiEBERT Model Card](https://huggingface.co/siebert/sentiment-roberta-large-english), this checkpoint stacks **24 Transformer layers** with a `1,024`-dimensional hidden state and `16` attention heads per layer, and was fine-tuned jointly across **15 diverse sentiment datasets** (spanning movie reviews, product reviews, tweets, and service feedback) to maximize cross-domain generalization.

## 3. Model Parameter and Specification Comparison (`66M` vs. `355M`)

The table below summarizes the structural, parametric, and training differences between the Initial Champion and the Challenger:

| Architectural Attribute | Model A — Initial Champion ([`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)) | Model B — Challenger ([`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english)) | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **Base Backbone** | [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) | [`FacebookAI/roberta-large`](https://huggingface.co/FacebookAI/roberta-large) | Distilled compact student vs. deep large-scale encoder |
| **Total Parameters** | **~66 Million (`66.9M`)** | **~355 Million (`355.4M`)** | Model B has **~5.3x** more learnable parameters than Model A |
| **Transformer Encoder Layers** | `6` layers | `24` layers | Model B performs `4x` more sequential self-attention blocks |
| **Hidden Dimension ($d_{\text{model}}$)** | `768` | `1,024` | Higher capacity per token embedding in Model B |
| **Attention Heads per Layer** | `12` heads (`64` dim/head) | `16` heads (`64` dim/head) | Finer multi-subspace syntactic and semantic tracking in Model B |
| **Feed-Forward Inner Dim ($d_{\text{ff}}$)** | `3,072` | `4,096` | Larger non-linear transformation capacity per layer in Model B |
| **Tokenizer & Vocabulary Size** | WordPiece (Uncased, `30,522` tokens) | Byte-Level BPE (Case-Sensitive, `50,265` tokens) | Model B retains capitalization cues and avoids unknown tokens |
| **Maximum Sequence Length** | `512` tokens | `512` tokens | Identical context window enforced (`max_length = 512`) |
| **Fine-Tuning Training Data** | Single corpus: SST-2 (short movie phrases/sentences) | Multi-corpus: 15 English sentiment datasets (reviews, tweets, texts) | Model B generalizes better across long, multi-paragraph texts |
| **Approximate FP32 Weight Size** | `~268 MB` | `~1.42 GB` | Model A requires `~5.3x` less disk and RAM storage |

## 4. Advantages and Disadvantages of Each Model

### 4.1 Model A: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (66M Parameters)

**Advantages:**
* **Fast CPU Inference & High Throughput:** With only 6 layers and 66M parameters ([DistilBERT Model Card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)), forward-pass matrix multiplications complete roughly `3x to 5x` faster on CPU (`device = -1`) than a 24-layer large model.
* **Low Memory Footprint:** Occupying around `268 MB` in 32-bit floating point, it loads quickly into memory and fits comfortably inside constrained Docker containers, microservices, or student laptops.
* **Strong Baseline Accuracy on Direct Sentiment:** Knowledge distillation preserves ~97% of BERT-base capabilities ([DistilBERT Docs](https://huggingface.co/docs/transformers/en/model_doc/distilbert)), yielding high precision and recall on reviews with explicit positive or negative adjectives.

**Disadvantages:**
* **Sentence-Level Training Bias (SST-2):** Because [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) was fine-tuned on short SST-2 snippets, multi-paragraph [IMDb reviews](https://huggingface.co/datasets/stanfordnlp/imdb) that recap a dark movie plot before praising the acting can confuse the classifier.
* **Loss of Case Sensitivity:** Lowercasing all text strips out emotional emphasis communicated via ALL-CAPS words.
* **Reduced Representational Depth:** With 6 layers, it has less capacity to resolve double negation, sarcasm, or contrasting clauses (*"I expected to hate this film, and the first ten minutes were rough, yet it turned into a masterpiece"*).

### 4.2 Model B: `siebert/sentiment-roberta-large-english` (355M Parameters)

**Advantages:**
* **Broad Cross-Domain Generalization:** Fine-tuned on 15 diverse English sentiment datasets ([SiEBERT Model Card](https://huggingface.co/siebert/sentiment-roberta-large-english)), it handles long-form narrative reviews, mixed sentiments, and varied writing styles more reliably than single-corpus SST-2 models.
* **Superior Contextual Reasoning:** Stacking 24 layers with 1,024-dimensional hidden states ([`FacebookAI/roberta-large`](https://huggingface.co/FacebookAI/roberta-large)) allows the model to weigh the overarching conclusion of a review against earlier plot descriptions.
* **Case-Aware Subword Representation:** Preserves capitalization and punctuation nuances through its 50,265-token Byte-Level BPE vocabulary.

**Disadvantages:**
* **High Computational and Latency Overhead:** With `355M` parameters, each forward pass executes roughly `5.3x` more floating-point operations, noticeably increasing average latency (`avg_latency_ms`) when running on CPU (`device = -1`).
* **Heavy Storage and RAM Demands:** The `~1.42 GB` checkpoint slows down cold-start container initialization, model downloading, and [`mlflow.pyfunc.log_model()`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html) serialization.
* **Risk of Violating Real-Time SLAs:** Without GPU acceleration or quantization, a 355M-parameter model may breach strict latency budgets in synchronous web APIs.

## 5. Comparison of Intended Use Cases

| Deployment Scenario | Recommended Model | Technical Justification |
| :--- | :--- | :--- |
| **Real-Time Synchronous Web / REST APIs (CPU-Hosted)** | **Model A ([DistilBERT 66M](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english))** | Strict tail-latency SLAs (e.g., `< 100 ms` per request on CPU) favor DistilBERT's 6-layer architecture and small memory footprint. |
| **High-Accuracy Offline Batch Analytics & Auditing** | **Model B ([SiEBERT RoBERTa-Large 355M](https://huggingface.co/siebert/sentiment-roberta-large-english))** | Overnight or asynchronous batch scoring of customer feedback prioritizes maximum Macro-F1 and nuanced understanding over millisecond latency. |
| **Multi-Domain Sentiment Aggregation (Tweets, Reviews, Tickets)** | **Model B ([SiEBERT RoBERTa-Large 355M](https://huggingface.co/siebert/sentiment-roberta-large-english))** | SiEBERT's 15-dataset training mixture prevents domain-shift degradation when inputs vary in length and tone. |
| **Edge Devices, Serverless Functions & CI/CD Smoke Tests** | **Model A ([DistilBERT 66M](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english))** | Fast cold-start loading (`~268 MB` vs. `~1.42 GB`) avoids serverless memory limits and keeps CI pipelines quick. |
| **GPU-Accelerated Production Microservices** | **Model B ([SiEBERT RoBERTa-Large 355M](https://huggingface.co/siebert/sentiment-roberta-large-english))** | Parallel GPU tensor cores compress RoBERTa-Large latency into a few milliseconds, unlocking its higher accuracy with minimal speed penalty. |

## 6. Why Macro-F1 Is Our Primary Evaluation Metric

In `evaluate.py`, we log five classification quality metrics (`accuracy`, `precision`, `recall`, `f1`, and `macro_f1`) alongside two speed metrics (`avg_latency_ms` and `total_inference_s`), and we designate **Macro-F1 (`macro_f1`)** — computed via [`sklearn.metrics.f1_score(actual, predictions, average="macro")`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) — as our **primary decision metric**:

### 6.1 Mathematical Definition

For binary sentiment classification with classes $c \in \{0\text{ (NEGATIVE)}, 1\text{ (POSITIVE)}\}$, the per-class $F_1$ score is the harmonic mean of that class's Precision ($P_c$) and Recall ($R_c$):

$$F_{1,c} = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c} = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$

**Macro-F1** is the unweighted arithmetic mean of the $F_1$ scores across both classes ([scikit-learn `f1_score` Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)):

$$\text{Macro-F1} = \frac{F_{1,\text{NEGATIVE (0)}} + F_{1,\text{POSITIVE (1)}}}{2}$$

### 6.2 Why Macro-F1 Beats Raw Accuracy and Standard Binary F1

1. **Treats Positive and Negative Classes Equally Under Sample Imbalance:** Our 100-review sample (`data/eval_inputs.csv`, drawn with `SEED = 42` from [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb)) contains **53 Negative (`0`)** and **47 Positive (`1`)** examples. Raw `accuracy` weights every example equally, which can allow performance on the slightly larger class (`NEGATIVE`) to mask errors on the smaller class (`POSITIVE`). Macro-F1 gives `50%` weight to $F_{1,\text{NEGATIVE}}$ and `50%` weight to $F_{1,\text{POSITIVE}}$.
2. **Avoids Positive-Class Tunnel Vision of Binary F1:** By default, standard binary `f1_score(actual, predictions)` (`average="binary"`, `pos_label=1`) only measures precision and recall on the **Positive (`1`)** class ([scikit-learn `f1_score` Reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)). In sentiment monitoring (such as detecting unhappy customers or negative movie reception), accurately identifying **Negative (`0`)** reviews with high precision and recall is just as critical as identifying Positive ones.
3. **Penalizes Polarity Bias (Over-Predicting One Class):** Because the harmonic mean punishes extreme trade-offs between Precision and Recall, a model that aggressively predicts `POSITIVE` (achieving high positive recall but poor negative recall) suffers a sharp drop in $F_{1,\text{NEGATIVE}}$ and consequently a lower Macro-F1.
4. **Pairs Cleanly with Our Margin (`0.01`) and Latency Guardrail (`5.0x`):** Requiring $\text{Macro-F1}_B - \text{Macro-F1}_A \ge 0.01$ ensures that the Challenger (`355M` parameters) must achieve a balanced, meaningful gain across *both* sentiment polarities — while staying within `5.0x` of Model A's inference latency — before replacing the lighter Initial Champion (`66M` parameters).

## 7. Git Commit Command for This Deliverable

After placing `MODEL_COMPARISON.md` in the root of your `1MS26CI405-mlflow-champion-challenger` project folder, stage and commit it in the PyCharm Terminal:

```bash
git add MODEL_COMPARISON.md
git commit -m "docs: add detailed champion challenger model comparison"
```
