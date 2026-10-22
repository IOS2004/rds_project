# LightGCN: Implementation, Evaluation, Layer Adaptation, and Cosine Contrastive Loss (CCL)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An educational, modular, and mathematically faithful PyTorch implementation of the seminal LightGCN paper and its contrastive learning adaptation from the SimpleX paper:

1. **LightGCN Paper:**
   > **“LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation”**  
   > *Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang (SIGIR 2020)*  
   > [arXiv:2002.02126](https://arxiv.org/abs/2002.02126) | Local PDF: [`2002.02126v4.pdf`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/2002.02126v4.pdf)

2. **SimpleX Paper:**
   > **“SimpleX: A Simple and Strong Baseline for Collaborative Filtering”**  
   > *Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, and Xiuqiang He (CIKM 2021)*  
   > [arXiv:2109.12613](https://arxiv.org/abs/2109.12613) | Local PDF: [`simplex.pdf`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/simplex.pdf)

This repository contains three distinct, rigorously evaluated stages:
1. **Stage A — Faithful Baseline:** An exact implementation of the original LightGCN model with linear bipartite propagation, symmetric degree normalization, uniform multi-layer aggregation ($1 / (K+1)$), dot-product recommendation scoring, and mini-batch BPR loss with layer-0 embedding L2 regularization.
2. **Stage B — Experimental Layer-Weight Adaptation (`AdaptiveLightGCN`):** An architectural extension that replaces the baseline's fixed uniform layer aggregation weights with trainable softmax parameters ($\alpha_k = \operatorname{softmax}(\theta)_k$).
3. **Stage C — Cosine Contrastive Loss Variant (`LightGCN-CCL`):** An adaptation based on the SimpleX paper (Section 3.1 & Section 4.5). It keeps the existing LightGCN graph propagation and uniform layer aggregation encoder intact, replacing the pairwise BPR loss with Cosine Contrastive Loss (CCL) and scaling negative sampling from 1 up to many unobserved items ($|N| \ge 50$) with cosine similarity ranking.

---

## Table of Contents

- [1. Recommender Systems Concepts: A Beginner's Guide](#1-recommender-systems-concepts-a-beginners-guide)
  - [Explicit Ratings vs. Implicit Feedback](#explicit-ratings-vs-implicit-feedback)
  - [The User-Item Bipartite Graph](#the-user-item-bipartite-graph)
  - [Degree-Based Graph Normalization](#degree-based-graph-normalization)
  - [Embedding Propagation (Graph Convolution)](#embedding-propagation-graph-convolution)
  - [Layer Combination and Oversmoothing](#layer-combination-and-oversmoothing)
  - [Bayesian Personalized Ranking (BPR) Loss](#bayesian-personalized-ranking-bpr-loss)
  - [Cosine Contrastive Loss (CCL) & Negative Sampling Ratios](#cosine-contrastive-loss-ccl--negative-sampling-ratios)
  - [Top-N Ranking Metrics: Recall@K and NDCG@K](#top-n-ranking-metrics-recallk-and-ndcgk)
- [2. Mathematical Formulations & Paper Mapping](#2-mathematical-formulations--paper-mapping)
- [3. Project Architecture](#3-project-architecture)
- [4. Setup and Quickstart](#4-setup-and-quickstart)
- [5. Running Unit Tests](#5-running-unit-tests)
- [6. Training the Baseline LightGCN](#6-training-the-baseline-lightgcn)
- [7. Stage B: AdaptiveLightGCN and Controlled Experiments](#7-stage-b-adaptivelightgcn-and-controlled-experiments)
- [8. Stage C: LightGCN-CCL (SimpleX Cosine Contrastive Loss)](#8-stage-c-lightgcn-ccl-simplex-cosine-contrastive-loss)
- [9. Three-Way Experimental Comparison & Empirical Results](#9-three-way-experimental-comparison--empirical-results)
- [10. Scientific Discussion & Published Paper Context](#10-scientific-discussion--published-paper-context)

---

## 1. Recommender Systems Concepts: A Beginner's Guide

### Explicit Ratings vs. Implicit Feedback
- **Explicit Ratings:** Users directly tell the platform how much they liked an item (e.g., 1 to 5 stars on MovieLens). While informative, explicit ratings are scarce.
- **Implicit Feedback:** Real-world interaction data consists of implicit behaviors: clicks, purchases, views, or page visits. In implicit feedback, we only observe positive signals ($R_{ui} = 1$). An unobserved entry ($R_{uj} = 0$) does not mean user $u$ hates item $j$; it usually means the user was simply unaware of it.

In this project, explicit ratings (MovieLens) are converted to implicit feedback by treating ratings $\ge 3$ as positive interactions ($R_{ui} = 1$), while unobserved entries form candidate items for negative sampling and ranking.

### The User-Item Bipartite Graph
Users and items are represented as nodes in an undirected bipartite graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- $\mathcal{V} = \mathcal{U} \cup \mathcal{I}$, with $M$ users and $N$ items ($|\mathcal{V}| = M + N$).
- An undirected edge $(u, i) \in \mathcal{E}$ exists whenever user $u$ interacted with item $i$.
- The adjacency matrix $A \in \mathbb{R}^{(M+N) \times (M+N)}$ is:
$$A = \begin{pmatrix} 0 & R \\ R^T & 0 \end{pmatrix}$$

### Degree-Based Graph Normalization
Standard GCN normalizes the adjacency matrix symmetrically:
$$\tilde{A} = D^{-1/2} A D^{-1/2}$$
For an edge between user $u$ and item $i$, the normalized propagation coefficient is:
$$\tilde{A}_{u, M+i} = \frac{1}{\sqrt{|N_u|} \sqrt{|N_i|}}$$
where $|N_u|$ and $|N_i|$ are node degrees. Isolated nodes have $D_{ii}^{-1/2} = 0$.

### Embedding Propagation (Graph Convolution)
LightGCN iteratively propagates information along graph edges without feature transformation matrices or non-linear activations:
- **Layer 0:** Initial embeddings $E^{(0)} = [P; Q]$.
- **Layer $k+1$:** $E^{(k+1)} = \tilde{A} E^{(k)}$.

### Layer Combination and Oversmoothing
To mitigate oversmoothing across graph convolution layers, representations are combined across all layers:
$$E^* = \sum_{k=0}^K \alpha_k E^{(k)}$$
- **Baseline LightGCN:** Uniform weights $\alpha_k = \frac{1}{K + 1}$.
- **AdaptiveLightGCN:** Softmax learnable weights $\alpha_k = \operatorname{softmax}(\theta)_k$.
- **LightGCN-CCL:** Uniform weights $\alpha_k = \frac{1}{K + 1}$ (retaining the exact LightGCN encoder).

### Bayesian Personalized Ranking (BPR) Loss
Pairwise ranking objective:
$$\mathcal{L}_{\text{BPR}} = -\frac{1}{|B|}\sum_{(u, i, j) \in B} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \frac{\lambda}{2|B|} \sum_{(u, i, j) \in B} \left(\|e_u^{(0)}\|^2 + \|e_i^{(0)}\|^2 + \|e_j^{(0)}\|^2\right)$$
where $\hat{y}_{ui} = (e_u^*)^T e_i^*$ is the unnormalized inner product.

### Cosine Contrastive Loss (CCL) & Negative Sampling Ratios
The SimpleX paper (CIKM 2021) argues that loss functions and negative sampling ratios are bottlenecks in collaborative filtering. CCL addresses two major issues in BPR:
1. **Popularity and Magnitude Bias:** Dot products depend on vector magnitudes. CCL uses cosine similarity $y_{ui} = \cos(e_u^*, e_i^*)$ to isolate directional preference.
2. **Easy Negatives Overwhelming Gradients:** Standard BPR treats all negatives equally. In large negative pools, most negatives are uninformative. CCL uses a margin $m \in [0, 1]$ and $\text{ReLU}(y_{uj} - m)$ to zero-out easy negatives and penalize only hard negatives.
3. **Imbalance Penalty:** With $|N| \gg 1$ (e.g. 50 to 1000 negatives per positive), negative loss is averaged over $|N|$ and scaled by hyperparameter $w$:
$$\mathcal{L}_{CCL}(u, i) = (1 - y_{ui}) + \frac{w}{|N|} \sum_{j \in N} \max(0, y_{uj} - m)$$

### Top-N Ranking Metrics: Recall@K and NDCG@K
At evaluation time:
1. Observed training interactions are masked to $-\infty$.
2. Candidate scores are computed (dot product for BPR, cosine similarity for CCL).
3. The model selects the top-$K$ items $\mathcal{R}_u(K)$ and evaluates:
   $$\text{Recall@K}(u) = \frac{|\mathcal{R}_u(K) \cap \mathcal{T}_u|}{|\mathcal{T}_u|}, \quad \text{NDCG@K}(u) = \frac{\text{DCG@K}(u)}{\text{IDCG@K}(u)}$$

---

## 2. Mathematical Formulations & Paper Mapping

| Component | Paper Formula | Code Location |
| :--- | :--- | :--- |
| **Bipartite Graph** | $A = \begin{pmatrix} 0 & R \\ R^T & 0 \end{pmatrix}$ | [`src/graph.py:build_sparse_adjacency`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/graph.py) |
| **Degree Normalization** | $\tilde{A} = D^{-1/2} A D^{-1/2}$ | [`src/graph.py:build_normalized_adjacency`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/graph.py) |
| **Propagation** | $E^{(k+1)} = \tilde{A} E^{(k)}$ | [`src/models/lightgcn.py:forward_propagation`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/models/lightgcn.py) |
| **Uniform Aggregation** | $E^* = \frac{1}{K+1}\sum_{k=0}^K E^{(k)}$ | [`src/models/lightgcn.py:forward_propagation`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/models/lightgcn.py) |
| **Adaptive Aggregation** | $E^* = \sum_{k=0}^K \text{softmax}(\theta)_k E^{(k)}$ | [`src/models/adaptive_lightgcn.py:forward_propagation`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/models/adaptive_lightgcn.py) |
| **BPR Recommendation Score** | $\hat{y}_{ui} = (e_u^*)^T e_i^*$ | [`src/models/lightgcn.py:predict`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/models/lightgcn.py) |
| **BPR Pairwise Loss + Reg** | $-\ln\sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \frac{\lambda}{2}\sum\|e^{(0)}\|^2$ | [`src/loss.py:BPRLoss`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/loss.py) |
| **CCL Loss (SimpleX)** | $(1 - y_{ui}) + \frac{w}{\|N\|}\sum_{j \in N}\max(0, y_{uj} - m)$ | [`src/loss.py:CosineContrastiveLoss`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/loss.py) |
| **CCL Cosine Scoring** | $y_{ui} = \frac{(e_u^*)^T e_i^*}{\|e_u^*\|_2 \|e_i^*\|_2}$ | [`src/models/lightgcn_ccl.py:predict`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/models/lightgcn_ccl.py) |
| **Multi-Negative Sampler** | $j \sim \{0, \dots, N-1\} \setminus N_u$, $\|N\| \ge 1$ | [`src/sampler.py:BPRSampler`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/sampler.py) |
| **Metrics Evaluation** | Full candidate Recall@K, NDCG@K (dot or cosine) | [`src/metrics.py:evaluate_model`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/src/metrics.py) |

---

## 3. Project Architecture

```text
rds_project/
├── 2002.02126v4.pdf              # Original LightGCN paper (He et al., SIGIR 2020)
├── simplex.pdf                   # SimpleX research paper (Mao et al., CIKM 2021)
├── docs/
│   └── paper_notes.md            # In-depth notes, mathematical formulations, and symbols
├── data/
│   ├── download_data.py          # Data download and split pipeline (MovieLens-100K / Synthetic)
│   └── ml-100k/                  # Preprocessed dataset splits (train.csv, val.csv, test.csv)
├── src/
│   ├── __init__.py
│   ├── dataset.py                # ID remapping, deduplication, leak-free stratified splitting
│   ├── graph.py                  # Undirected bipartite graph and D^{-1/2} A D^{-1/2} normalization
│   ├── loss.py                   # BPRLoss and CosineContrastiveLoss (CCL) with L2 regularization
│   ├── sampler.py                # Uniform negative sampler supporting 1 and multiple negatives
│   ├── metrics.py                # Top-N ranking metrics supporting dot-product and cosine scoring
│   ├── trainer.py                # Optimization loop for BPR and CCL, model selection, checkpointing
│   └── models/
│       ├── __init__.py           # Exports LightGCN, AdaptiveLightGCN, LightGCNCCL
│       ├── lightgcn.py           # Baseline LightGCN (uniform aggregation, BPR dot-scoring)
│       ├── adaptive_lightgcn.py  # Stage B: AdaptiveLightGCN (learnable softmax layer weights)
│       └── lightgcn_ccl.py       # Stage C: LightGCN-CCL (SimpleX CCL objective and cosine scoring)
├── tests/
│   ├── test_dataset.py           # ID mapping, deduplication, non-leakage tests
│   ├── test_graph.py             # Symmetry, degrees, normalization on toy graphs
│   ├── test_model.py             # Single-step propagation and uniform aggregation math
│   ├── test_adaptive_model.py    # Uniform init, gradient flow to theta, baseline equivalence
│   ├── test_loss.py              # BPR ranking ordering and L2 reg calculation
│   ├── test_ccl_loss.py          # Hand-calculated CCL math, margin threshold, averaging, gradient flow
│   ├── test_sampler.py           # Single- and multi-negative item exclusion, shapes, edge cases
│   ├── test_lightgcn_ccl.py      # End-to-end toy training, cosine evaluation, finite metrics
│   ├── test_metrics.py           # Hand-calculated Recall/NDCG and candidate exclusion
│   └── test_reproducibility.py   # Seed reproducibility verification
├── scripts/
│   ├── train_baseline.py         # Trains baseline LightGCN model (BPR)
│   ├── train_adaptive.py         # Trains AdaptiveLightGCN model (BPR)
│   ├── train_lightgcn_ccl.py     # Trains LightGCN-CCL model (CCL with multi-negative sampling)
│   └── run_comparison.py         # Controlled multi-seed 3-way experiment comparison and plotting
├── experiments/                  # Checkpoints, summaries, comparison CSVs, and plots
├── requirements.txt              # Project dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 4. Setup and Quickstart

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA)

### Installation
```bash
pip install -r requirements.txt
```

### Dataset Preparation
```bash
# Download and prepare MovieLens-100K (ratings >= 3 as positive, 80/10/10 split)
python data/download_data.py --dataset ml-100k
```

---

## 5. Running Unit Tests

A comprehensive suite of **32 unit tests** verifies all mathematical formulations, graph normalization, loss mechanics, candidate masking, negative sampling protocols, and seed reproducibility:

```bash
pytest tests/ -v
```

All 32 tests execute in under 5 seconds:
* `tests/test_adaptive_model.py`: 3 tests (uniform initialization, gradient flow to $\theta$, baseline equivalence).
* `tests/test_ccl_loss.py`: 7 tests (hand-calculated numerical example, below-margin zero penalty, above-margin linear penalty, $|N|$ averaging, numerical stability on zero vectors, gradient flow to initial embeddings, regularization scaling).
* `tests/test_dataset.py`: 3 tests (ID mapping, deduplication, non-leakage).
* `tests/test_graph.py`: 3 tests (bipartite construction, symmetry, degree normalization).
* `tests/test_lightgcn_ccl.py`: 3 tests (forward pass/prediction, training loop step, cosine evaluation finiteness).
* `tests/test_loss.py`: 2 tests (BPR loss ordering and L2 reg).
* `tests/test_metrics.py`: 2 tests (hand-calculated Recall/NDCG, candidate exclusion).
* `tests/test_model.py`: 2 tests (manual propagation step, recommendation scoring).
* `tests/test_reproducibility.py`: 1 test (seed reproducibility).
* `tests/test_sampler.py`: 6 tests (single/multi-negative exclusion, shapes, invalid counts, dense-user edge case, batch iteration).

---

## 6. Training the Baseline LightGCN

Train the original LightGCN model with BPR loss:
```bash
python scripts/train_baseline.py \
    --data_dir data/ml-100k \
    --embedding_dim 64 \
    --num_layers 3 \
    --learning_rate 0.001 \
    --reg_weight 1e-4 \
    --epochs 30 \
    --eval_every 5 \
    --save_dir experiments/baseline_ml100k
```

---

## 7. Stage B: AdaptiveLightGCN and Controlled Experiments

### What is AdaptiveLightGCN?
AdaptiveLightGCN replaces the fixed uniform layer weights:
$$\alpha_k = \frac{1}{K + 1}$$
with trainable softmax parameters:
$$\alpha_k = \operatorname{softmax}(\theta)_k = \frac{\exp(\theta_k)}{\sum_{j=0}^K \exp(\theta_j)}$$
initialized at $\theta_k = 0$ so that $\alpha_k = 1 / (K + 1)$ at initialization.

Train the adaptive variant:
```bash
python scripts/train_adaptive.py \
    --data_dir data/ml-100k \
    --embedding_dim 64 \
    --num_layers 3 \
    --epochs 30 \
    --save_dir experiments/adaptive_ml100k
```

---

## 8. Stage C: LightGCN-CCL (SimpleX Cosine Contrastive Loss)

### Background & Motivation
In **"SimpleX: A Simple and Strong Baseline for Collaborative Filtering"** (CIKM 2021), Mao et al. showed that collaborative filtering performance is heavily bottlenecked by the choice of loss function and negative sampling ratio, rather than the complexity of the interaction encoder. In Section 4.5 and Table 3 of the SimpleX paper, the authors demonstrated that applying Cosine Contrastive Loss (CCL) directly to LightGCN substantially improves its performance over BPR.

### Scope Restriction
- **LightGCN Encoder Retained:** This variant does *not* introduce the SimpleX user behavior aggregation layer or any other neural architecture. It keeps the exact LightGCN bipartite propagation $E^{(k+1)} = \tilde{A}E^{(k)}$ and uniform layer aggregation $E^* = \frac{1}{K+1}\sum E^{(k)}$.
- **Loss and Sampling Adapted:** The training objective is switched from pairwise BPR to Cosine Contrastive Loss (CCL) paired with multi-negative sampling ($|N| \ge 50$) and cosine similarity evaluation.

### Mathematical Formulation
For each user $u$, observed positive item $i$, and sampled negative item set $N$:
$$\mathcal{L}_{\text{CCL}}(u, i) = (1 - y_{ui}) + \frac{w}{|N|} \sum_{j \in N} \max(0, y_{uj} - m)$$
where:
* $y_{ui} = \cos(e_u^*, e_i^*) = \frac{(e_u^*)^T e_i^*}{\max(\|e_u^*\|_2 \|e_i^*\|_2, \epsilon)}$ is the cosine similarity between the final aggregated user and positive item embeddings.
* $y_{uj} = \cos(e_u^*, e_j^*)$ is the cosine similarity between the final user embedding and each sampled negative item embedding.
* $m$ is the margin threshold (e.g., $0.8$). Negatives with similarity $y_{uj} \le m$ incur zero penalty.
* $w$ is the negative loss weight (e.g., $10.0$).
* $|N|$ is the number of negative samples per positive interaction.

### Documented Embedding Regularization
Embedding L2 regularization is applied to the initial layer-0 embeddings ($e_u^{(0)}, e_i^{(0)}, e_j^{(0)}$) of the active mini-batch. To maintain a scale-consistent penalty per training interaction regardless of $|N|$, the negative energy is normalized by $|N|$:
$$\mathcal{L}_{\text{reg}} = \frac{\lambda}{2|B|} \sum_{b=1}^{|B|} \left( \|e_{u,b}^{(0)}\|_2^2 + \|e_{i,b}^{(0)}\|_2^2 + \frac{1}{|N|} \sum_{j \in N_b} \|e_{j,b}^{(0)}\|_2^2 \right)$$
When $|N| = 1$, this formulation is identical to standard LightGCN BPR regularization.

### Negative Sampling Protocol
- Negatives are sampled uniformly from unobserved items: $\mathcal{I} \setminus \mathcal{N}_u$.
- Observed training positive interactions are strictly excluded.
- For $|N| > 1$, sampling across candidate draws is with replacement to guarantee validity even if a user has fewer eligible unobserved items than $|N|$.
- Batch sampling is performed on-the-fly to ensure $O(|B| \times |N|)$ memory scaling without allocating large full-dataset arrays.

### Training LightGCN-CCL
```bash
python scripts/train_lightgcn_ccl.py \
    --data_dir data/ml-100k \
    --embedding_dim 64 \
    --num_layers 3 \
    --learning_rate 0.001 \
    --reg_weight 1e-4 \
    --batch_size 1024 \
    --epochs 30 \
    --ccl_margin 0.8 \
    --ccl_negative_weight 10.0 \
    --num_negatives 50 \
    --score_type cosine \
    --save_dir experiments/lightgcn_ccl
```

---

## 9. Three-Way Experimental Comparison & Empirical Results

### Running the Controlled 3-Way Comparison
Run the experiment comparing Baseline LightGCN, AdaptiveLightGCN, and LightGCN-CCL on identical splits and random seeds:
```bash
python scripts/run_comparison.py \
    --data_dir data/ml-100k \
    --seeds 42 123 \
    --epochs 15 \
    --embedding_dim 64 \
    --num_layers 3 \
    --ccl_margin 0.8 \
    --ccl_negative_weight 10.0 \
    --ccl_num_negatives 50 \
    --ccl_score_type cosine \
    --save_dir experiments/comparison
```

This generates:
- `raw_seed_metrics.csv`: Individual seed evaluation metrics.
- `comparison_summary.csv`: Aggregated means and standard deviations.
- `comparison_results.json`: Complete machine-readable experimental log and configuration.
- `metric_comparison.png`: 3-way grouped bar chart with error bars.
- `learned_weights.png`: Adaptive layer weights visualization across seeds.

### 9.1 Controlled Empirical Results (MovieLens-100K)
The table below reports the results from our controlled run across random seeds `42` and `123` on MovieLens-100K (15 epochs, full catalog candidate ranking with training item masking):

| Model Variant | Training Objective | Negative Ratio ($|N|$) | Scoring Function | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline LightGCN** | BPR Loss | 1 | Dot Product | $0.1794 \pm 0.0006$ | $0.2693 \pm 0.0025$ | $0.1616 \pm 0.0006$ | $0.1849 \pm 0.0011$ |
| **AdaptiveLightGCN** | BPR Loss | 1 | Dot Product | $0.1826 \pm 0.0018$ | $0.2851 \pm 0.0037$ | $0.1729 \pm 0.0020$ | $0.2001 \pm 0.0016$ |
| **LightGCN-CCL** | **Cosine Contrastive (CCL)** | **50** | **Cosine Similarity** | **$0.2270 \pm 0.0061$** | **$0.3470 \pm 0.0053$** | **$0.2221 \pm 0.0035$** | **$0.2535 \pm 0.0032$** |
| **Relative $\Delta$ (CCL vs Baseline)** | — | — | — | **$+26.5\%$** | **$+28.9\%$** | **$+37.4\%$** | **$+37.1\%$** |

*(Values can be verified in [`experiments/comparison/comparison_summary.csv`](file:///c:/Users/om%20sahu/OneDrive/Desktop/rds_project/experiments/comparison/comparison_summary.csv).)*

---

## 10. Scientific Discussion & Published Paper Context

### Comparison with Published Literature (SimpleX Paper Table 3)
In Section 4.5 and Table 3 of the SimpleX paper (*SimpleX: A Simple and Strong Baseline for Collaborative Filtering*, CIKM 2021), Mao et al. reported the following results on larger benchmark datasets when replacing BPR with CCL on LightGCN:

| Dataset | Model Variant | Recall@20 | NDCG@20 | Published Source |
| :--- | :--- | :--- | :--- | :--- |
| **Amazon-Books** | LightGCN (BPR) | $0.0411$ | $0.0315$ | SIGIR 2020 / SimpleX Table 3 |
| **Amazon-Books** | LightGCN-CCL (CCL) | **$0.0528$** ($+28.5\%$) | **$0.0416$** ($+32.1\%$) | SimpleX Table 3 (CIKM 2021) |
| **Yelp2018** | LightGCN (BPR) | $0.0649$ | $0.0530$ | SIGIR 2020 / SimpleX Table 3 |
| **Yelp2018** | LightGCN-CCL (CCL) | **$0.0669$** ($+3.1\%$) | **$0.0554$** ($+4.5\%$) | SimpleX Table 3 (CIKM 2021) |

### Key Insights:
1. **Consistency across Benchmarks:** Our empirical results on MovieLens-100K ($+28.9\%$ Recall@20, $+37.1\%$ NDCG@20) align with the published gains observed by Mao et al. on Amazon-Books ($+28.5\%$ Recall@20, $+32.1\%$ NDCG@20).
2. **Why CCL Outperforms BPR with LightGCN:**
   - **Direction vs. Magnitude:** BPR with dot products tends to inflate norms for high-degree nodes. By constraining representations to directional cosine alignment, CCL prevents popularity bias from skewing recommendations.
   - **Hard Negative Filtering:** The margin threshold $m$ ensures that uninformative easy negatives (similarity $\le m$) produce zero loss. The optimizer focuses parameter updates exclusively on hard negatives ($y_{uj} > m$).
   - **Sample Density:** Sampling 50 to 1000 negatives gives the contrastive loss a denser view of the unobserved space without degenerating into uniform noise.
