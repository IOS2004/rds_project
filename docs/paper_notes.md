# Mathematical and Implementation Notes: LightGCN

> Paper: **“LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation”**  
> Authors: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang  
> Conference: ACM SIGIR 2020  
> arXiv: [2002.02126](https://arxiv.org/abs/2002.02126)

---

## 1. Problem Formulation and Assumptions

- **Setting:** Implicit-feedback collaborative filtering for recommendation.
- **Input:** A set of users $\mathcal{U} = \{u_1, u_2, \dots, u_M\}$ of size $M$, and a set of items $\mathcal{I} = \{i_1, i_2, \dots, i_N\}$ of size $N$.
- **Interactions:** Binary interaction matrix $R \in \mathbb{R}^{M \times N}$, where:
  $$R_{ui} = \begin{cases} 1, & \text{if user } u \text{ has interacted with item } i \\ 0, & \text{otherwise} \end{cases}$$
- **Key Observation / Hypothesis:** In standard GCNs for collaborative filtering (such as NGCF), feature transformation matrices ($W_1, W_2$) and nonlinear activations ($\sigma$) add burdensome computation, make training difficult, and do not improve (and often degrade) recommendation accuracy. Removing them simplifies GCN to its core mechanism: **linear neighborhood smoothing on the graph**.

---

## 2. Bipartite Graph Construction and Adjacency

The user-item interaction graph is an undirected bipartite graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V} = \mathcal{U} \cup \mathcal{I}$ with $|\mathcal{V}| = M + N$.

### Adjacency Matrix
$$A = \begin{pmatrix} 0 & R \\ R^T & 0 \end{pmatrix} \in \mathbb{R}^{(M+N) \times (M+N)}$$

- The top-left $M \times M$ block is zero (no user-user explicit edges).
- The bottom-right $N \times N$ block is zero (no item-item explicit edges).
- $R \in \mathbb{R}^{M \times N}$ represents the user-item observed edges.

### Degree Matrix
Diagonal matrix $D \in \mathbb{R}^{(M+N) \times (M+N)}$ where:
$$D_{ii} = \sum_{j=1}^{M+N} A_{ij}$$
- For a user node $u \in \{0, \dots, M-1\}$: $D_{uu} = |N_u|$, the number of items interacted by $u$.
- For an item node $i \in \{0, \dots, N-1\}$: $D_{M+i, M+i} = |N_i|$, the number of users who interacted with $i$.
- **Handling Isolated Nodes:** If a node has degree 0 (no interactions in training split), $D_{ii} = 0$. To prevent division by zero during $D^{-1/2}$, we set $D_{ii}^{-1/2} = 0$ for isolated nodes.

### Symmetric Degree-Normalized Adjacency Matrix
$$\tilde{A} = D^{-1/2} A D^{-1/2}$$

For an edge between user $u$ and item $i$:
$$\tilde{A}_{u, M+i} = \frac{1}{\sqrt{|N_u|}\sqrt{|N_i|}}, \quad \tilde{A}_{M+i, u} = \frac{1}{\sqrt{|N_i|}\sqrt{|N_u|}}$$

**Implementation Component:** `src/graph.py` (`build_normalized_adjacency`).

---

## 3. Embedding Propagation (Light Graph Convolution - LGC)

### 3.1 Initial Embeddings (Layer 0)
Trainable parameter matrices:
- $P \in \mathbb{R}^{M \times d}$: Layer 0 user ID embeddings.
- $Q \in \mathbb{R}^{N \times d}$: Layer 0 item ID embeddings.
- Concatenated embedding matrix at layer 0:
  $$E^{(0)} = \begin{pmatrix} P \\ Q \end{pmatrix} \in \mathbb{R}^{(M+N) \times d}$$
  where $d$ is the embedding dimension (default $d = 64$).
- Initialized using Xavier uniform initialization (Glorot uniform).

### 3.2 Propagation Equation
For layer $k \in \{0, 1, \dots, K-1\}$:
$$E^{(k+1)} = \tilde{A} E^{(k)} = \left(D^{-1/2} A D^{-1/2}\right) E^{(k)}$$

In node-wise form:
$$e_u^{(k+1)} = \sum_{i \in N_u} \frac{1}{\sqrt{|N_u|}\sqrt{|N_i|}} e_i^{(k)}$$
$$e_i^{(k+1)} = \sum_{u \in N_i} \frac{1}{\sqrt{|N_i|}\sqrt{|N_u|}} e_u^{(k)}$$

**Key Properties:**
- No feature transformations (no weight matrices $W$).
- No nonlinear activations (no ReLU, LeakyReLU, or Sigmoid).
- No self-loops in $A$ ($A_{ii} = 0$). Self-connection is subsumed by the subsequent layer combination.

**Implementation Component:** `src/models/lightgcn.py` (`LightGCN.forward_propagation`).

---

## 4. Multi-Layer Combination (Final Aggregation)

After computing embeddings at layers $k = 0, 1, \dots, K$, the final representation for each node is a weighted sum over all layers:

$$E^* = \sum_{k=0}^{K} \alpha_k E^{(k)}$$

For user $u$ and item $i$:
$$e_u^* = \sum_{k=0}^{K} \alpha_k e_u^{(k)}, \quad e_i^* = \sum_{k=0}^{K} \alpha_k e_i^{(k)}$$

### 4.1 Baseline LightGCN (Uniform Fixed Weights)
$$\alpha_k = \frac{1}{K + 1} \quad \text{for all } k \in \{0, 1, \dots, K\}$$

### 4.2 Stage B: AdaptiveLightGCN (Learnable Layer Weights)
Layer weights are parameterized by unconstrained trainable logits $\theta \in \mathbb{R}^{K+1}$:
$$\alpha_k = \operatorname{softmax}(\theta)_k = \frac{\exp(\theta_k)}{\sum_{j=0}^K \exp(\theta_j)}$$
- Initialization: $\theta_k = 0$ for all $k$, which yields $\alpha_k = \frac{1}{K+1}$ at initialization.
- Optimization: Updated via backpropagation alongside embedding parameters.

**Implementation Components:**
- Baseline: `src/models/lightgcn.py` (`LightGCN`).
- Adaptive: `src/models/adaptive_lightgcn.py` (`AdaptiveLightGCN`).

---

## 5. Prediction Score Calculation

The recommendation ranking score for user $u$ and item $i$ is calculated via inner product (dot product) of their final embeddings:

$$\hat{y}_{ui} = \langle e_u^*, e_i^* \rangle = (e_u^*)^T e_i^* = \sum_{f=1}^d e_{uf}^* e_{if}^*$$

For full candidate ranking across all items:
$$\hat{\mathbf{y}}_u = Q^* (e_u^*)^T \in \mathbb{R}^N$$
where $Q^* \in \mathbb{R}^{N \times d}$ is the final item embedding matrix.

**Implementation Component:** `src/models/lightgcn.py` (`LightGCN.predict`, `LightGCN.predict_all`).

---

## 6. Training Objective (Bayesian Personalized Ranking - BPR)

The model is trained using the pairwise BPR loss:

$$\mathcal{L}_{BPR} = -\sum_{u=1}^M \sum_{i \in N_u} \sum_{j \notin N_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|E^{(0)}\|_F^2$$

where:
- $u$ is a target user.
- $i \in N_u$ is an observed (positive) training item for user $u$.
- $j \notin N_u$ is an unobserved (negative) item sampled uniformly from $\mathcal{I} \setminus N_u$.
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid activation function.
- $\lambda$ is the L2 regularization coefficient.

### Mini-Batch Loss Formulation
For a mini-batch $\mathcal{B} = \{(u, i, j)\}$ of size $B$:
$$\mathcal{L}_{\text{batch}} = \frac{1}{|\mathcal{B}|} \sum_{(u, i, j) \in \mathcal{B}} -\ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \cdot \frac{1}{2 |\mathcal{B}|} \sum_{(u, i, j) \in \mathcal{B}} \left( \|e_u^{(0)}\|_2^2 + \|e_i^{(0)}\|_2^2 + \|e_j^{(0)}\|_2^2 \right)$$

*Important detail:* Following the paper and its official reference implementation, L2 regularization applies **only to layer-0 embeddings** $E^{(0)}$ of the entities sampled in the batch, not the propagated embeddings $E^*$.

**Implementation Component:** `src/loss.py` (`BPRLoss`).

---

## 7. Negative Sampling and Candidate Exclusion

1. **Training Sampling:** For each positive user-item pair $(u, i)$ in the training set, sample a random item $j \in \{0, \dots, N-1\}$ such that $(u, j) \notin \text{PositiveInteractions}_{\text{train}}(u)$.
2. **Evaluation Candidate Exclusion:** In accordance with standard ranking protocols in recommendation, any item interacted with by user $u$ during **training** is masked out (assigned score $-\infty$) so it is never ranked among top-$K$ recommendations at test time.

**Implementation Components:**
- Sampler: `src/sampler.py` (`BPRSampler`).
- Evaluation: `src/metrics.py` (`evaluate_ranking`).

---

## 8. Evaluation Metrics

### 8.1 Recall@K
For user $u$ with ground-truth test item set $\mathcal{T}_u$:
$$\text{Recall@K}(u) = \frac{|\mathcal{R}_u(K) \cap \mathcal{T}_u|}{|\mathcal{T}_u|}$$
where $\mathcal{R}_u(K)$ is the set of top-$K$ recommended items.

### 8.2 NDCG@K
$$\text{DCG@K}(u) = \sum_{p=1}^K \frac{2^{r_p} - 1}{\log_2(p+1)} = \sum_{p=1}^K \frac{r_p}{\log_2(p+1)}$$
where $r_p \in \{0, 1\}$ indicates whether the item at position $p$ is in $\mathcal{T}_u$.
$$\text{IDCG@K}(u) = \sum_{p=1}^{\min(|\mathcal{T}_u|, K)} \frac{1}{\log_2(p+1)}$$
$$\text{NDCG@K}(u) = \frac{\text{DCG@K}(u)}{\text{IDCG@K}(u)} \quad (\text{defined as } 0 \text{ if } \text{IDCG} = 0)$$

Reported metrics are averaged over all evaluated test users:
$$\text{Recall@K} = \frac{1}{|\mathcal{U}_{\text{test}}|} \sum_{u \in \mathcal{U}_{\text{test}}} \text{Recall@K}(u), \quad \text{NDCG@K} = \frac{1}{|\mathcal{U}_{\text{test}}|} \sum_{u \in \mathcal{U}_{\text{test}}} \text{NDCG@K}(u)$$

**Implementation Component:** `src/metrics.py` (`compute_recall_at_k`, `compute_ndcg_at_k`, `evaluate_user_recommendations`).

---

## 9. Paper Settings vs Implementation Details

| Parameter / Setting | Paper Specification | Implementation Setting / Notes |
| :--- | :--- | :--- |
| **Embedding Dimension ($d$)** | 64 | 64 (configurable) |
| **Number of Layers ($K$)** | 1 to 4; default optimal $K=3$ | Configurable (default $K=3$) |
| **Layer Weights ($\alpha_k$)** | Uniform $1/(K+1)$ | Uniform for baseline; softmax for adaptive |
| **Initialization** | Xavier uniform | `torch.nn.init.xavier_uniform_` |
| **Optimizer** | Adam (learning rate 0.001) | `torch.optim.Adam(lr=0.001)` |
| **L2 Regularization ($\lambda$)** | Searched in $\{10^{-6}, 10^{-5}, \dots, 10^{-2}\}$; optimal $\approx 10^{-4}$ | $10^{-4}$ default |
| **Batch Size** | 1024 (2048 for Amazon-Book) | 1024 (configurable) |
| **Candidate Protocol** | Full item space minus user's training interactions | Full item ranking, excluding train set items |
| **Datasets** | Gowalla, Yelp2018, Amazon-Book | MovieLens-100K / MovieLens-1M & Gowalla support |
| **Explicit to Implicit Conversion** | Not needed for Gowalla/Yelp (already implicit) | For MovieLens: rating $\ge 3$ (or all observed) marked positive |

---

## 10. Key Author Notes on Learnable Layer Weights ($\alpha_k$)

In Section 3.3 of the paper, the authors explicitly state:
> *"Moreover, it is technically viable to also learn the layer combination coefficients $\{\alpha_k\}_{k=0}^K$, or parameterize them with an attention network. However, we find that learning $\alpha$ on training data does not lead improvement. This is probably because the training data does not contain sufficient signal to learn good $\alpha$ that can generalize to unknown data. We have also tried to learn $\alpha$ from validation data... performance is slightly improved (less than 1%)."*

This is crucial context for **Stage B**:
- In our experimental study, we implement `AdaptiveLightGCN` using softmax-parameterized layer weights $\alpha_k = \operatorname{softmax}(\theta)_k$.
- We empirically test whether training-set gradient updates improve, degrade, or match uniform weights.
- We report empirical findings objectively and analyze the trajectories of the learned layer weights $\theta_k$.
