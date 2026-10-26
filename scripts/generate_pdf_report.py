"""
generate_pdf_report.py
======================
Generates a comprehensive, publication-grade HTML report and compiles it into a PDF
document using headless Microsoft Edge.

The report covers:
  1. Executive Summary & Core Contributions
  2. Theoretical Foundation of Collaborative Filtering & LightGCN
  3. System Architecture & End-to-End Pipeline
  4. The Evaluated Model Variants (Baseline, Adaptive, CCL, Softmax Contrastive)
  5. Loss Function Mechanics & Mathematical Formulations
  6. Controlled Empirical Benchmark Results (MovieLens-100K across seeds 42 and 123)
  7. Loss Parameter Ablation & Hyperparameter Sensitivity Curves
  8. Representation Geometry: Hinge Dead Zone vs. Angular Separation Margin
  9. Literature Context (Comparison with Published SimpleX Table 3)
 10. Execution, Verification Suite (33 Tests), & Interactive Top-K Inference

Output:
  - docs/project_report.html (Self-contained HTML report with embedded base64 graphics)
  - docs/LightGCN_Project_Report.pdf (Full PDF document)
  - LightGCN_Project_Report.pdf (Root-level copy)
"""

import os
import sys
import base64
import shutil
import subprocess

def get_base64_image(image_path: str) -> str:
    """Reads an image file and converts it into a base64 Data URI."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{data}"

def build_html_report() -> str:
    """Constructs the complete publication-grade HTML report."""
    # Encode images as base64 data URIs
    img_arch = get_base64_image("docs/system_architecture.png")
    img_bar = get_base64_image("experiments/comparison/metric_comparison.png")
    img_radar = get_base64_image("experiments/comparison/radar_and_relative_gain.png")
    img_weights = get_base64_image("experiments/comparison/learned_weights.png")
    img_deadzone = get_base64_image("experiments/comparison/loss_mechanisms_and_dead_zone.png")
    img_ablation = get_base64_image("experiments/comparison/ablation_tuning_curves.png")
    img_kde = get_base64_image("experiments/comparison/representation_similarity_density.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LightGCN: Comprehensive Implementation, Evaluation, Layer Adaptation, and Contrastive Learning</title>
  <style>
    /* Global Page & Print Rules */
    @page {{
      size: A4;
      margin: 16mm 14mm 16mm 14mm;
      @bottom-right {{
        content: "Page " counter(page);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 8.5pt;
        color: #64748b;
      }}
      @bottom-left {{
        content: "LightGCN & Contrastive Learning Technical Report";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 8.5pt;
        color: #64748b;
      }}
    }}

    *, *:before, *:after {{
      box-sizing: border-box;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 9.5pt;
      line-height: 1.5;
      color: #1e293b;
      background-color: #ffffff;
      margin: 0;
      padding: 0;
    }}

    /* Page Breaks */
    .page-break {{
      page-break-before: always;
    }}

    .no-break {{
      page-break-inside: avoid;
    }}

    /* Typography */
    h1, h2, h3, h4 {{
      color: #0f172a;
      font-weight: 700;
      margin-top: 0;
      page-break-after: avoid;
    }}

    h1 {{
      font-size: 22pt;
      line-height: 1.25;
      margin-bottom: 6px;
      letter-spacing: -0.5px;
    }}

    .subtitle {{
      font-size: 11.5pt;
      color: #475569;
      font-weight: 400;
      line-height: 1.4;
      margin-bottom: 12px;
    }}

    h2 {{
      font-size: 13pt;
      border-bottom: 2px solid #2563eb;
      padding-bottom: 4px;
      margin-top: 20px;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      letter-spacing: -0.3px;
    }}

    h2 .sec-num {{
      background: #2563eb;
      color: #ffffff;
      font-size: 9.5pt;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 4px;
      margin-right: 8px;
    }}

    h3 {{
      font-size: 10.5pt;
      color: #1e3a8a;
      margin-top: 14px;
      margin-bottom: 6px;
      letter-spacing: -0.2px;
    }}

    p {{
      margin: 0 0 8px 0;
      text-align: justify;
    }}

    /* Badges & Meta Pills */
    .meta-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 14px;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 8pt;
      font-weight: 600;
    }}

    .badge-blue {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }}
    .badge-green {{ background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }}
    .badge-purple {{ background: #faf5ff; color: #7e22ce; border: 1px solid #e9d5ff; }}
    .badge-slate {{ background: #f8fafc; color: #334155; border: 1px solid #e2e8f0; }}

    /* Callout & Summary Boxes */
    .summary-box {{
      background: #f8fafc;
      border-left: 4px solid #2563eb;
      border-radius: 0 6px 6px 0;
      padding: 10px 14px;
      margin-bottom: 16px;
    }}

    .summary-title {{
      font-weight: 700;
      color: #1e40af;
      margin-bottom: 4px;
      font-size: 9.5pt;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    /* Equation Boxes */
    .eq-box {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 14px;
      margin: 8px 0 10px 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      page-break-inside: avoid;
    }}

    .eq-formula {{
      font-family: "Cambria Math", "Latin Modern Math", Georgia, serif;
      font-size: 10.5pt;
      color: #0f172a;
    }}

    .eq-num {{
      font-size: 8.5pt;
      color: #64748b;
      font-weight: 600;
      font-family: inherit;
    }}

    /* Matrix inline styling */
    .matrix-inline {{
      display: inline-flex;
      align-items: center;
      vertical-align: middle;
      font-family: inherit;
      margin: 0 2px;
    }}
    .matrix-bracket {{
      font-size: 1.35em;
      font-weight: 300;
      line-height: 1;
    }}
    .matrix-cells {{
      display: inline-grid;
      grid-template-columns: auto auto;
      gap: 2px 8px;
      text-align: center;
      padding: 0 3px;
      font-size: 0.95em;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 14px 0;
      font-size: 8.5pt;
      page-break-inside: avoid;
    }}

    th, td {{
      padding: 5px 8px;
      text-align: left;
      border: 1px solid #cbd5e1;
    }}

    th {{
      background-color: #0f172a;
      color: #ffffff;
      font-weight: 600;
      font-size: 8pt;
      letter-spacing: 0.2px;
    }}

    tr:nth-child(even) {{
      background-color: #f8fafc;
    }}

    tr.highlight-row {{
      background-color: #ecfdf5 !important;
      font-weight: 600;
    }}

    td.num, th.num {{
      text-align: right;
    }}

    td.center, th.center {{
      text-align: center;
    }}

    .pill-win {{
      background: #059669;
      color: #ffffff;
      padding: 1px 6px;
      border-radius: 10px;
      font-size: 7.5pt;
      font-weight: 700;
    }}

    /* Figures & Cards */
    .figure-card {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px;
      margin: 10px 0 12px 0;
      text-align: center;
      page-break-inside: avoid;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }}

    .figure-card img {{
      max-width: 100%;
      height: auto;
      max-height: 235px;
      border-radius: 4px;
      display: block;
      margin: 0 auto;
    }}

    .figure-caption {{
      margin-top: 6px;
      font-size: 8pt;
      color: #475569;
      text-align: left;
      line-height: 1.35;
      border-top: 1px solid #f1f5f9;
      padding-top: 5px;
    }}

    .figure-caption strong {{
      color: #0f172a;
    }}

    /* Code Snippets */
    pre, code {{
      font-family: "Cascadia Code", "Fira Code", Consolas, "Courier New", monospace;
    }}

    code {{
      background: #f1f5f9;
      color: #0f172a;
      padding: 1px 4px;
      border-radius: 3px;
      font-size: 8pt;
    }}

    pre {{
      background: #0f172a;
      color: #f8fafc;
      padding: 9px 12px;
      border-radius: 6px;
      font-size: 8pt;
      line-height: 1.4;
      overflow-x: auto;
      margin: 8px 0 12px 0;
      page-break-inside: avoid;
    }}

    pre code {{
      background: transparent;
      color: inherit;
      padding: 0;
      border-radius: 0;
    }}

    /* Two-column layout */
    .row {{
      display: flex;
      gap: 12px;
      margin: 8px 0;
      page-break-inside: avoid;
    }}

    .col {{
      flex: 1;
    }}

    /* Code syntax coloring */
    .kw {{ color: #93c5fd; font-weight: 600; }}
    .str {{ color: #86efac; }}
    .com {{ color: #94a3b8; font-style: italic; }}
    .num-c {{ color: #fde047; }}
    .fn {{ color: #67e8f9; }}

    /* Key Finding Callouts */
    .takeaway-list {{
      margin: 6px 0 10px 0;
      padding-left: 16px;
    }}

    .takeaway-list li {{
      margin-bottom: 4px;
      font-size: 9pt;
    }}

    .takeaway-list strong {{
      color: #0f172a;
    }}
  </style>
</head>
<body>

  <!-- COVER / HEADER -->
  <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 14px;">
    <h1>LightGCN: Implementation, Evaluation, Layer Adaptation & Contrastive Learning</h1>
    <div class="subtitle">
      A Scientific Investigation into Graph Convolutional Collaborative Filtering, Adaptive Layer Aggregation, and Gradient Dynamics in Contrastive Objectives
    </div>
    <div class="meta-bar">
      <span class="badge badge-blue">PyTorch 2.0+</span>
      <span class="badge badge-green">33 Unit Tests Passed</span>
      <span class="badge badge-purple">MovieLens-100K Benchmark</span>
      <span class="badge badge-slate">5 Model Variants & Multi-Seed Protocol</span>
    </div>
  </div>

  <!-- EXECUTIVE SUMMARY -->
  <div class="summary-box no-break">
    <div class="summary-title">Executive Summary</div>
    <p>
      This technical report documents the complete mathematical formulation, modular software engineering, empirical verification, and architectural extensions of <strong>LightGCN</strong> (He et al., SIGIR 2020) and its contrastive learning adaptations inspired by <strong>SimpleX</strong> (Mao et al., CIKM 2021).
      Through systematic multi-seed controlled experimentation on MovieLens-100K, we demonstrate that collaborative filtering performance is fundamentally bottlenecked by the choice of <em>loss objective</em> and <em>negative sampling ratio</em> rather than graph encoder complexity:
    </p>
    <ul style="margin: 4px 0 0 0; padding-left: 16px; font-size: 9pt;">
      <li><strong>Baseline LightGCN (BPR, 1 negative):</strong> Achieves <code>0.2693</code> Recall@20 and <code>0.1849</code> NDCG@20.</li>
      <li><strong>AdaptiveLightGCN (Trainable Softmax Aggregation):</strong> Learns non-uniform propagation weights favoring intermediate layers (k=1, 2), improving Recall@20 to <code>0.2851</code> (<strong>+5.9%</strong> relative gain).</li>
      <li><strong>LightGCN-CCL (Hinge Cosine Contrastive Loss, 50 negatives):</strong> Scales negative sampling with directional cosine scoring, boosting Recall@20 to <code>0.3464</code> (<strong>+28.6%</strong> relative gain over baseline).</li>
      <li><strong>LightGCN-Softmax (Sampled InfoNCE Contrastive Loss, &tau;=0.15):</strong> Eliminates the 81.5% zero-gradient dead zone of hinge loss via continuous, smooth competitive repulsion, achieving <code>0.3693</code> Recall@20 and <code>0.2658</code> NDCG@20 (<strong>+37.1%</strong> and <strong>+43.8%</strong> relative gains over baseline LightGCN).</li>
    </ul>
  </div>

  <!-- SECTION 1: THEORETICAL FOUNDATION -->
  <h2><span class="sec-num">1</span> Theoretical Foundations: Recommender Systems on Graphs</h2>
  
  <p>
    Collaborative filtering platforms record binary user-item interactions (clicks, purchases, ratings &ge; 3), representing observed positive feedback. Unobserved entries do not imply negative preferences; they primarily represent unexposed candidates.
  </p>

  <h3>1.1 User-Item Bipartite Graph & Degree Normalization</h3>
  <p>
    Let <em>M</em> denote the number of users and <em>N</em> the number of items. The interaction topology forms an undirected bipartite graph 
    <em>G = (V, E)</em> with total vertices <em>|V| = M + N</em>. The bipartite adjacency matrix <em>A</em> is defined as:
  </p>

  <div class="eq-box">
    <div class="eq-formula">
      A = 
      <span class="matrix-inline">
        <span class="matrix-bracket">[</span>
        <span class="matrix-cells">
          <span>0</span> <span>R</span>
          <span>R<sup>T</sup></span> <span>0</span>
        </span>
        <span class="matrix-bracket">]</span>
      </span>
      &isin; &reals;<sup>(M+N) &times; (M+N)</sup>
    </div>
    <div class="eq-num">(Eq. 1)</div>
  </div>

  <p>
    Standard Graph Convolutional Networks (GCN) suffer from node-degree scale divergence. LightGCN applies symmetric degree-based normalization:
  </p>

  <div class="eq-box">
    <div class="eq-formula">
      A&#771; = D<sup>-1/2</sup> A D<sup>-1/2</sup>, &emsp; A&#771;<sub>u, M+i</sub> = 1 / (&radic;|N<sub>u</sub>| &radic;|N<sub>i</sub>|)
    </div>
    <div class="eq-num">(Eq. 2)</div>
  </div>

  <h3>1.2 Light Graph Convolution (LGC) & Mitigation of Oversmoothing</h3>
  <p>
    Conventional GCNs (e.g., NGCF) employ heavy weight transformation matrices <em>W</em> and non-linear activations &sigma;(x). The seminal insight of LightGCN is that node features in collaborative filtering are simply categorical IDs with no inherent semantic features. Feature transformation and non-linearities degrade recommendation performance and make training unstable.
    LightGCN strips away all non-linearities and parameter matrices, distilling graph convolution down to pure linear neighborhood smoothing:
  </p>

  <div class="eq-box">
    <div class="eq-formula">
      E<sup>(0)</sup> = [P; Q] &isin; &reals;<sup>(M+N) &times; d</sup>, &emsp; E<sup>(k+1)</sup> = A&#771; E<sup>(k)</sup>
    </div>
    <div class="eq-num">(Eq. 3)</div>
  </div>

  <p>
    To alleviate oversmoothing (the tendency of node embeddings to collapse into identical averages across deeper layers), multi-layer representations are combined into a final representation:
  </p>

  <div class="eq-box">
    <div class="eq-formula">
      E<sup>*</sup> = &sum;<sub>k=0</sub><sup>K</sup> &alpha;<sub>k</sub> E<sup>(k)</sup>, &emsp; 
      e<sub>u</sub><sup>*</sup> = &sum;<sub>k=0</sub><sup>K</sup> &alpha;<sub>k</sub> e<sub>u</sub><sup>(k)</sup>, &emsp;
      e<sub>i</sub><sup>*</sup> = &sum;<sub>k=0</sub><sup>K</sup> &alpha;<sub>k</sub> e<sub>i</sub><sup>(k)</sup>
    </div>
    <div class="eq-num">(Eq. 4)</div>
  </div>

  <!-- SECTION 2: SYSTEM ARCHITECTURE -->
  <div class="page-break"></div>
  <h2><span class="sec-num">2</span> End-to-End System Architecture</h2>
  
  <p>
    The modular framework maps directly from raw implicit interactions to graph construction, multi-layer propagation, scoring heads, and loss optimization:
  </p>

  <div class="figure-card">
    <img src="{img_arch}" alt="End-to-End System Architecture">
    <div class="figure-caption">
      <strong>Figure 1: End-to-end architectural system diagram.</strong> 
      (1) Raw interaction tuples are preprocessed with leak-free stratified splitting. 
      (2) Sparse bipartite adjacency matrix is normalized symmetrically. 
      (3) Light Graph Convolution performs linear smoothing over <em>K=3</em> propagation layers. 
      (4) Layer combination aggregates representations using uniform weights (Baseline/CCL) or trainable softmax logits (Adaptive). 
      (5) Scoring heads evaluate dot-product or cosine similarity against full item candidate pools with training item masking. 
      (6) Objectives optimize mini-batches using BPR, Hinge CCL, or Cosine Softmax Contrastive Loss.
    </div>
  </div>

  <h3>2.1 Code-to-Paper Component Mapping</h3>
  <table>
    <thead>
      <tr>
        <th>Framework Component</th>
        <th>Mathematical Formulation</th>
        <th>Repository Module & Implementation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Bipartite Adjacency</strong></td>
        <td>A = [0, R; R<sup>T</sup>, 0]</td>
        <td><code>src/graph.py</code>: <code>build_sparse_adjacency</code></td>
      </tr>
      <tr>
        <td><strong>Degree Normalization</strong></td>
        <td>A&#771; = D<sup>-1/2</sup> A D<sup>-1/2</sup></td>
        <td><code>src/graph.py</code>: <code>build_normalized_adjacency</code></td>
      </tr>
      <tr>
        <td><strong>LGC Propagation</strong></td>
        <td>E<sup>(k+1)</sup> = A&#771; E<sup>(k)</sup></td>
        <td><code>src/models/lightgcn.py</code>: <code>forward_propagation</code></td>
      </tr>
      <tr>
        <td><strong>Uniform Layer Combination</strong></td>
        <td>E<sup>*</sup> = (1/(K+1)) &sum; E<sup>(k)</sup></td>
        <td><code>src/models/lightgcn.py</code>: <code>forward_propagation</code></td>
      </tr>
      <tr>
        <td><strong>Adaptive Layer Aggregation</strong></td>
        <td>E<sup>*</sup> = &sum; softmax(&theta;)<sub>k</sub> E<sup>(k)</sup></td>
        <td><code>src/models/adaptive_lightgcn.py</code>: <code>forward_propagation</code></td>
      </tr>
      <tr>
        <td><strong>Pairwise BPR Loss + Reg</strong></td>
        <td>-ln &sigma;(y&#770;<sub>ui</sub> - y&#770;<sub>uj</sub>) + (&lambda;/2)&sum;||e<sup>(0)</sup>||<sup>2</sup></td>
        <td><code>src/loss.py</code>: <code>BPRLoss</code></td>
      </tr>
      <tr>
        <td><strong>Hinge Cosine Contrastive (CCL)</strong></td>
        <td>(1 - y<sub>ui</sub>) + (w/|N|)&sum; max(0, y<sub>uj</sub> - m)</td>
        <td><code>src/loss.py</code>: <code>CosineContrastiveLoss</code></td>
      </tr>
      <tr>
        <td><strong>Cosine Softmax Contrastive (InfoNCE)</strong></td>
        <td>-ln [exp(cos/&tau;) / &sum; exp(cos/&tau;)]</td>
        <td><code>src/loss.py</code>: <code>CosineSoftmaxLoss</code></td>
      </tr>
      <tr>
        <td><strong>Cosine Scoring Head</strong></td>
        <td>y<sub>ui</sub> = (e<sub>u</sub><sup>*</sup> &middot; e<sub>i</sub><sup>*</sup>) / (||e<sub>u</sub><sup>*</sup>|| &middot; ||e<sub>i</sub><sup>*</sup>||)</td>
        <td><code>src/models/lightgcn_ccl.py</code>: <code>predict_all_for_users</code></td>
      </tr>
      <tr>
        <td><strong>Multi-Negative Sampler</strong></td>
        <td>j &sim; I \ N<sub>u</sub> with |N| &ge; 50</td>
        <td><code>src/sampler.py</code>: <code>BPRSampler</code></td>
      </tr>
      <tr>
        <td><strong>Candidate Masking & Top-K Eval</strong></td>
        <td>Mask train items (-&infin;); Recall@K & NDCG@K</td>
        <td><code>src/metrics.py</code>: <code>evaluate_user_recommendations</code></td>
      </tr>
    </tbody>
  </table>

  <!-- SECTION 3: LOSS MECHANICS & THE DEAD ZONE -->
  <div class="page-break"></div>
  <h2><span class="sec-num">3</span> Loss Function Mechanics & The Hinge Dead Zone</h2>

  <p>
    The project evaluates three fundamentally distinct loss mechanisms to train the LightGCN encoder:
  </p>

  <h3>3.1 Bayesian Personalized Ranking (BPR) Loss</h3>
  <p>
    Standard LightGCN relies on BPR pairwise loss with inner-product scoring y&#770;<sub>ui</sub> = (e<sub>u</sub><sup>*</sup>)<sup>T</sup> e<sub>i</sub><sup>*</sup>.
    Only a single negative item <em>j</em> is sampled per positive item <em>i</em>:
  </p>
  <div class="eq-box">
    <div class="eq-formula">
      L<sub>BPR</sub> = - (1/|B|) &sum;<sub>(u,i,j)</sub> ln &sigma;(y&#770;<sub>ui</sub> - y&#770;<sub>uj</sub>) + (&lambda;/2|B|) &sum; (||e<sub>u</sub><sup>(0)</sup>||<sup>2</sup> + ||e<sub>i</sub><sup>(0)</sup>||<sup>2</sup> + ||e<sub>j</sub><sup>(0)</sup>||<sup>2</sup>)
    </div>
    <div class="eq-num">(Eq. 5)</div>
  </div>

  <h3>3.2 SimpleX Cosine Contrastive Loss (CCL)</h3>
  <p>
    Mao et al. (CIKM 2021) identified two critical flaws in BPR: dot products suffer from severe node popularity bias (unbounded embedding magnitudes), and 1 negative sample per positive provides insufficient coverage of the unobserved space. 
    CCL enforces directional alignment on the unit hypersphere and scales negative sampling (|N| &ge; 50):
  </p>
  <div class="eq-box">
    <div class="eq-formula">
      L<sub>CCL</sub>(u, i) = (1 - y<sub>ui</sub>) + (w / |N|) &sum;<sub>j &isin; N</sub> max(0, y<sub>uj</sub> - m)
    </div>
    <div class="eq-num">(Eq. 6)</div>
  </div>

  <h3>3.3 The Hinge Dead Zone & Continuous Softmax Contrastive Loss</h3>
  <p>
    While Hinge CCL dramatically outperforms BPR, representation diagnostics revealed a severe limitation: 
    <strong>81.5% of sampled negatives have similarity below the margin threshold (y<sub>uj</sub> &lt; 0.8)</strong>, yielding an exact zero gradient:
    <em>&part;L<sub>CCL</sub> / &part;y<sub>uj</sub> = 0</em>. Over four-fifths of the sampled negative items provide zero learning signal!
  </p>

  <div class="figure-card">
    <img src="{img_deadzone}" alt="Loss Mechanisms and Hinge Dead Zone">
    <div class="figure-caption">
      <strong>Figure 2: Theoretical gradient repulsion mechanisms & empirical dead zone distribution.</strong>
      <em>Left:</em> Repulsive gradient magnitude |&part;L/&part;y<sub>uj</sub>| plotted against negative sample cosine similarity. Hinge CCL exhibits a hard cutoff at margin <em>m</em> (zero gradient for all easy negatives). In contrast, Cosine Softmax provides smooth, continuous repulsive gradients across the entire similarity spectrum.
      <em>Right:</em> Empirical histogram of negative sample cosine similarities under trained LightGCN-CCL (m=0.8), proving that <strong>81.5% of negatives reside in the inactive dead zone</strong>.
    </div>
  </div>

  <p>
    To resolve the dead zone, we formulated <strong>Cosine Softmax Contrastive Loss</strong> (sampled InfoNCE):
  </p>

  <div class="eq-box">
    <div class="eq-formula">
      L<sub>softmax</sub> = - (1/|B|) &sum;<sub>(u,i) &isin; B</sub> ln [ exp(cos(e<sub>u</sub><sup>*</sup>, e<sub>i</sub><sup>*</sup>) / &tau;) / (exp(cos(e<sub>u</sub><sup>*</sup>, e<sub>i</sub><sup>*</sup>) / &tau;) + &sum;<sub>j &isin; N<sub>u</sub></sub> exp(cos(e<sub>u</sub><sup>*</sup>, e<sub>j</sub><sup>*</sup>) / &tau;)) ]
    </div>
    <div class="eq-num">(Eq. 7)</div>
  </div>

  <p>
    Under Cosine Softmax, the negative repulsive gradient is <em>-p<sub>uj</sub> / &tau;</em>. Because the softmax probability <em>p<sub>uj</sub> &gt; 0</em> everywhere, every single unobserved negative sample exerts continuous repulsive pressure, dynamically scaling with its hardness.
  </p>

  <!-- SECTION 4: EMPIRICAL BENCHMARK RESULTS -->
  <div class="page-break"></div>
  <h2><span class="sec-num">4</span> Controlled Multi-Seed Empirical Results (MovieLens-100K)</h2>

  <p>
    All five model variants were trained under identical conditions (15 epochs, batch size 1024, embedding dimension 64, 3 GCN propagation layers, Adam optimizer with lr=0.001) across multiple random seeds (42 and 123) on the official MovieLens-100K stratified split. Evaluation ranks all unobserved candidates across the entire item catalog with observed training items masked to -&infin;.
  </p>

  <table>
    <thead>
      <tr>
        <th>Model Variant</th>
        <th>Training Objective</th>
        <th class="center">Negatives (|N|)</th>
        <th>Scoring Metric</th>
        <th class="num">Recall@10</th>
        <th class="num">Recall@20</th>
        <th class="num">NDCG@10</th>
        <th class="num">NDCG@20</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>LightGCN (BPR)</strong></td>
        <td>Pairwise BPR</td>
        <td class="center">1</td>
        <td>Dot Product</td>
        <td class="num">0.1794 &plusmn; 0.0006</td>
        <td class="num">0.2693 &plusmn; 0.0026</td>
        <td class="num">0.1616 &plusmn; 0.0006</td>
        <td class="num">0.1849 &plusmn; 0.0011</td>
      </tr>
      <tr>
        <td><strong>AdaptiveLightGCN (BPR)</strong></td>
        <td>Pairwise BPR</td>
        <td class="center">1</td>
        <td>Dot Product</td>
        <td class="num">0.1826 &plusmn; 0.0018</td>
        <td class="num">0.2851 &plusmn; 0.0036</td>
        <td class="num">0.1729 &plusmn; 0.0020</td>
        <td class="num">0.2001 &plusmn; 0.0016</td>
      </tr>
      <tr>
        <td><strong>LightGCN-CCL (m=0.8)</strong></td>
        <td>Hinge CCL (m=0.8)</td>
        <td class="center">50</td>
        <td>Cosine Similarity</td>
        <td class="num">0.2264 &plusmn; 0.0060</td>
        <td class="num">0.3464 &plusmn; 0.0040</td>
        <td class="num">0.2223 &plusmn; 0.0035</td>
        <td class="num">0.2537 &plusmn; 0.0029</td>
      </tr>
      <tr>
        <td><strong>LightGCN-CCL (m=0.6)</strong></td>
        <td>Hinge CCL (m=0.6)</td>
        <td class="center">50</td>
        <td>Cosine Similarity</td>
        <td class="num">0.2317 &plusmn; 0.0073</td>
        <td class="num">0.3468 &plusmn; 0.0031</td>
        <td class="num">0.2233 &plusmn; 0.0045</td>
        <td class="num">0.2527 &plusmn; 0.0031</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>LightGCN-Softmax (&tau;=0.15)</strong></td>
        <td>Softmax Contrastive (&tau;=0.15)</td>
        <td class="center">50</td>
        <td>Cosine Similarity</td>
        <td class="num"><strong>0.2422 &plusmn; 0.0026</strong></td>
        <td class="num"><strong>0.3693 &plusmn; 0.0019</strong></td>
        <td class="num"><strong>0.2317 &plusmn; 0.0011</strong></td>
        <td class="num"><strong>0.2658 &plusmn; 0.0005</strong></td>
      </tr>
      <tr style="background: #f1f5f9; font-weight: 700;">
        <td>Relative &Delta; (Softmax vs Baseline)</td>
        <td colspan="3" class="center"><span class="pill-win">Dominant Winner</span></td>
        <td class="num">+35.0%</td>
        <td class="num">+37.1%</td>
        <td class="num">+43.4%</td>
        <td class="num">+43.8%</td>
      </tr>
    </tbody>
  </table>

  <div class="row">
    <div class="col">
      <div class="figure-card">
        <img src="{img_bar}" alt="5-Model Bar Comparison" style="max-height: 200px;">
        <div class="figure-caption">
          <strong>Figure 3: 5-Model Grouped Bar Comparison.</strong>
          Test Recall@10/20 and NDCG@10/20 with error bars across random seeds 42 and 123.
        </div>
      </div>
    </div>
    <div class="col">
      <div class="figure-card">
        <img src="{img_radar}" alt="Radar & Relative Gains" style="max-height: 200px;">
        <div class="figure-caption">
          <strong>Figure 4: Radar Profile & Relative Gains.</strong>
          Shows massive step gains moving from BPR &rarr; Adaptive &rarr; CCL &rarr; Softmax InfoNCE.
        </div>
      </div>
    </div>
  </div>

  <div class="figure-card">
    <img src="{img_weights}" alt="Learned Layer Weights" style="max-height: 200px;">
    <div class="figure-caption">
      <strong>Figure 5: Learned layer combination weights (&alpha;<sub>k</sub> = softmax(&theta;)<sub>k</sub>) for AdaptiveLightGCN across seeds.</strong>
      Notice that the network autonomously allocates higher weight to intermediate propagation layers (k=1, 2) rather than the raw layer-0 embedding, demonstrating that higher-order graph smoothing conveys superior collaborative signal.
    </div>
  </div>

  <!-- SECTION 5: ABLATIONS & REPRESENTATION GEOMETRY -->
  <div class="page-break"></div>
  <h2><span class="sec-num">5</span> Hyperparameter Ablations & Representation Geometry</h2>

  <p>
    We conducted targeted ablations across margin thresholds <em>m &isin; [0.4, 0.8]</em> in Hinge CCL and temperatures <em>&tau; &isin; [0.07, 0.15]</em> in Cosine Softmax Contrastive Loss:
  </p>

  <table>
    <thead>
      <tr>
        <th>Loss Formulation</th>
        <th class="center">Tuned Hyperparameter</th>
        <th class="num">Recall@10</th>
        <th class="num">Recall@20</th>
        <th class="num">NDCG@10</th>
        <th class="num">NDCG@20</th>
        <th class="center">Gain vs Baseline CCL</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>LightGCN-CCL (Hinge Baseline)</strong></td>
        <td class="center">Margin m = 0.8</td>
        <td class="num">0.2283</td>
        <td class="num">0.3489</td>
        <td class="num">0.2242</td>
        <td class="num">0.2560</td>
        <td class="center">Baseline</td>
      </tr>
      <tr>
        <td><strong>Margin Calibration</strong></td>
        <td class="center">Margin m = 0.6</td>
        <td class="num">0.2369</td>
        <td class="num">0.3490</td>
        <td class="num">0.2265</td>
        <td class="num">0.2549</td>
        <td class="center">+0.0%</td>
      </tr>
      <tr>
        <td><strong>Margin Calibration</strong></td>
        <td class="center">Margin m = 0.4</td>
        <td class="num">0.2365</td>
        <td class="num">0.3464</td>
        <td class="num">0.2249</td>
        <td class="num">0.2528</td>
        <td class="center">-0.7%</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>Cosine Softmax (Sampled InfoNCE)</strong></td>
        <td class="center"><strong>Temperature &tau; = 0.15</strong></td>
        <td class="num"><strong>0.2440</strong></td>
        <td class="num"><strong>0.3706</strong></td>
        <td class="num"><strong>0.2325</strong></td>
        <td class="num"><strong>0.2655</strong></td>
        <td class="center"><span class="pill-win">+6.2% rel (+2.17% abs)</span></td>
      </tr>
      <tr>
        <td><strong>Cosine Softmax (Sampled InfoNCE)</strong></td>
        <td class="center">Temperature &tau; = 0.10</td>
        <td class="num">0.2265</td>
        <td class="num">0.3478</td>
        <td class="num">0.2179</td>
        <td class="num">0.2508</td>
        <td class="center">-0.3%</td>
      </tr>
      <tr>
        <td><strong>Cosine Softmax (Sampled InfoNCE)</strong></td>
        <td class="center">Temperature &tau; = 0.07</td>
        <td class="num">0.2045</td>
        <td class="num">0.3196</td>
        <td class="num">0.1939</td>
        <td class="num">0.2258</td>
        <td class="center">-8.4%</td>
      </tr>
    </tbody>
  </table>

  <div class="figure-card">
    <img src="{img_ablation}" alt="Parameter Tuning Curves" style="max-height: 220px;">
    <div class="figure-caption">
      <strong>Figure 6: Dual-axis sensitivity tuning curves.</strong>
      <em>Left:</em> Hinge margin calibration sweep (m &isin; [0.4, 0.8]), indicating that adjusting margin alone cannot overcome the dead zone bottleneck.
      <em>Right:</em> Softmax temperature sensitivity curve (&tau; &isin; [0.07, 0.15]), identifying an optimal operating regime at &tau;=0.15 that strikes an ideal balance between gradient sharpness and optimization stability.
    </div>
  </div>

  <h3>5.1 Angular Separation on the Unit Hypersphere</h3>
  <p>
    We extracted user and item representations from the trained checkpoints on the test set and computed Kernel Density Estimations (KDE) of positive user-item similarity vs unobserved negative similarity:
  </p>

  <div class="figure-card">
    <img src="{img_kde}" alt="Representation Density KDE" style="max-height: 220px;">
    <div class="figure-caption">
      <strong>Figure 7: Empirical cosine similarity KDE distributions on the unit hypersphere.</strong>
      <em>Baseline & Adaptive:</em> Suffer from representation overlap and weak directional discrimination.
      <em>LightGCN-CCL (m=0.8):</em> Forces unobserved negatives to pile up immediately below the margin boundary at mean similarity +0.76 (separation margin &Delta; = 0.047).
      <em>LightGCN-Softmax (&tau;=0.15):</em> Drives unobserved negatives into a wide, uniform distribution with mean -0.07 while maintaining positive alignment at +0.22, establishing a <strong>massive angular separation margin of &Delta; = 0.290</strong>.
    </div>
  </div>

  <!-- SECTION 6: PUBLISHED LITERATURE & EXECUTION -->
  <div class="page-break"></div>
  <h2><span class="sec-num">6</span> Literature Context, Verification Suite & Quickstart</h2>

  <h3>6.1 Alignment with Published Literature (SimpleX CIKM 2021 Table 3)</h3>
  <p>
    In Table 3 of the SimpleX paper, Mao et al. observed that replacing BPR with CCL on LightGCN yielded enormous improvements on large benchmark datasets:
  </p>
  <table>
    <thead>
      <tr>
        <th>Benchmark Dataset</th>
        <th>Model & Loss Variant</th>
        <th class="num">Recall@20</th>
        <th class="num">NDCG@20</th>
        <th>Published Source</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Amazon-Books</strong></td>
        <td>LightGCN (BPR)</td>
        <td class="num">0.0411</td>
        <td class="num">0.0315</td>
        <td>He et al. (SIGIR 2020)</td>
      </tr>
      <tr>
        <td><strong>Amazon-Books</strong></td>
        <td>LightGCN-CCL (CCL)</td>
        <td class="num"><strong>0.0528 (+28.5%)</strong></td>
        <td class="num"><strong>0.0416 (+32.1%)</strong></td>
        <td>SimpleX Table 3 (CIKM 2021)</td>
      </tr>
      <tr>
        <td><strong>Yelp2018</strong></td>
        <td>LightGCN (BPR)</td>
        <td class="num">0.0649</td>
        <td class="num">0.0530</td>
        <td>He et al. (SIGIR 2020)</td>
      </tr>
      <tr>
        <td><strong>Yelp2018</strong></td>
        <td>LightGCN-CCL (CCL)</td>
        <td class="num"><strong>0.0669 (+3.1%)</strong></td>
        <td class="num"><strong>0.0554 (+4.5%)</strong></td>
        <td>SimpleX Table 3 (CIKM 2021)</td>
      </tr>
      <tr class="highlight-row">
        <td><strong>MovieLens-100K (Our Benchmark)</strong></td>
        <td><strong>LightGCN-Softmax (&tau;=0.15)</strong></td>
        <td class="num"><strong>0.3693 (+37.1%)</strong></td>
        <td class="num"><strong>0.2658 (+43.8%)</strong></td>
        <td>This Project (Controlled Multi-Seed)</td>
      </tr>
    </tbody>
  </table>

  <h3>6.2 Comprehensive Verification Suite (33 Unit Tests)</h3>
  <p>
    The codebase is protected by a complete suite of 33 unit tests executing in under 5 seconds:
  </p>
  <ul class="takeaway-list">
    <li><code>tests/test_graph.py</code> (3 tests): Bipartite adjacency structure, symmetry, and degree normalization D<sup>-1/2</sup> A D<sup>-1/2</sup>.</li>
    <li><code>tests/test_model.py</code> (2 tests): Exact linear graph convolution propagation mathematics and recommendation scoring.</li>
    <li><code>tests/test_adaptive_model.py</code> (3 tests): Softmax parameter initialization (&theta;=0), baseline equivalence, and gradient flow.</li>
    <li><code>tests/test_loss.py</code> (2 tests): Pairwise BPR margin ordering and layer-0 L2 regularization scaling.</li>
    <li><code>tests/test_ccl_loss.py</code> (8 tests): Hand-calculated numerical examples, zero-penalty below margin, linear penalty above margin, |N| normalization, numerical stability, and cosine softmax contrastive loss.</li>
    <li><code>tests/test_sampler.py</code> (6 tests): Negative sampling validity, strict train positive exclusion, dense-user edge cases.</li>
    <li><code>tests/test_metrics.py</code> (2 tests): Full-candidate ranking, candidate masking (-&infin;), hand-verified Recall@K and NDCG@K.</li>
    <li><code>tests/test_dataset.py</code> (3 tests): ID remapping, deduplication, and leak-free stratified splitting.</li>
    <li><code>tests/test_lightgcn_ccl.py</code> (3 tests): End-to-end toy forward pass, optimizer step, and finite evaluation metrics.</li>
    <li><code>tests/test_reproducibility.py</code> (1 test): Multi-seed determinism and exact reproducibility.</li>
  </ul>

  <h3>6.3 Execution Commands & Interactive Inference</h3>
  <pre><code><span class="com"># 1. Run full 33-test verification suite</span>
pytest tests/ -v

<span class="com"># 2. Run controlled 5-model multi-seed benchmark and auto-generate diagrams</span>
python scripts/run_comparison.py --data_dir data/ml-100k --seeds 42 123 --epochs 15

<span class="com"># 3. Standalone publication diagram generator (generates 7 figures)</span>
python scripts/generate_diagrams.py

<span class="com"># 4. Generate HTML and PDF technical report</span>
python scripts/generate_pdf_report.py</code></pre>

  <p>
    To generate personalized Top-5 recommendations for any user using the trained checkpoint:
  </p>
  <pre><code><span class="kw">import</span> torch
<span class="kw">from</span> src.dataset <span class="kw">import</span> InteractionData
<span class="kw">from</span> src.graph <span class="kw">import</span> build_torch_normalized_adjacency
<span class="kw">from</span> src.models.lightgcn_ccl <span class="kw">import</span> LightGCNCCL
<span class="kw">from</span> scripts.train_baseline <span class="kw">import</span> load_dataset

<span class="com"># 1. Load dataset and construct normalized adjacency</span>
data = load_dataset(<span class="str">"data/ml-100k"</span>)
norm_adj = build_torch_normalized_adjacency(data.num_users, data.num_items, data.train_pairs)

<span class="com"># 2. Instantiate model and load trained best checkpoint</span>
model = LightGCNCCL(num_users=data.num_users, num_items=data.num_items, embedding_dim=<span class="num-c">64</span>, num_layers=<span class="num-c">3</span>, score_type=<span class="str">"cosine"</span>)
ckpt = torch.load(<span class="str">"experiments/comparison/lightgcn_softmax_tau015_seed_42/best_model.pt"</span>, map_location=<span class="str">"cpu"</span>)
model.load_state_dict(ckpt[<span class="str">"model_state_dict"</span>])
model.eval()

<span class="com"># 3. Predict all candidate scores and mask seen training items</span>
<span class="kw">with</span> torch.no_grad():
    scores = model.predict_all_for_users(torch.tensor([<span class="num-c">0</span>]), norm_adj)[<span class="num-c">0</span>]
seen = data.train_user_positives.get(<span class="num-c">0</span>, set())
<span class="kw">if</span> seen: scores[list(seen)] = -float(<span class="str">"inf"</span>)

<span class="com"># 4. Retrieve Top-5 recommendation rankings</span>
top_scores, top_items = torch.topk(scores, <span class="num-c">5</span>)
<span class="kw">for</span> rank, (item_idx, score) <span class="kw">in</span> enumerate(zip(top_items.tolist(), top_scores.tolist()), <span class="num-c">1</span>):
    print(f<span class="str">"Rank {{rank}}: Item Index {{item_idx:4d}} | Cosine Score: {{score:.4f}}"</span>)</code></pre>

  <div style="text-align: center; margin-top: 20px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 8pt; color: #64748b;">
    LightGCN Technical Report &bull; Comprehensive Implementation & Benchmark &bull; PyTorch 2.0+ &bull; 2026
  </div>

</body>
</html>
"""
    return html

def convert_html_to_pdf(html_path: str, pdf_path: str):
    """Invokes headless Microsoft Edge to convert HTML into a clean PDF."""
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    ]
    browser_exe = None
    for p in edge_paths:
        if p and os.path.exists(p):
            browser_exe = p
            break

    if not browser_exe:
        raise RuntimeError("No headless browser (Edge/Chrome) found for PDF conversion.")

    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={abs_pdf}",
        "--no-pdf-header-footer",
        abs_html
    ]

    print(f"Running browser print-to-pdf using: {browser_exe}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error during browser execution: {res.stderr}")
        raise RuntimeError(f"Browser PDF generation failed with code {res.returncode}")

    if not os.path.exists(abs_pdf):
        raise FileNotFoundError(f"PDF was not created at {abs_pdf}")
    
    print(f"PDF successfully generated at {abs_pdf} (Size: {os.path.getsize(abs_pdf):,} bytes)")

def main():
    print("=" * 70)
    print("Generating Comprehensive Project HTML and PDF Technical Report")
    print("=" * 70)

    # 1. Build and save self-contained HTML
    html_content = build_html_report()
    html_path = os.path.join("docs", "project_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Self-contained HTML report written to {html_path} (Size: {len(html_content):,} chars)")

    # 2. Convert to PDF in docs/
    pdf_docs_path = os.path.join("docs", "LightGCN_Project_Report.pdf")
    convert_html_to_pdf(html_path, pdf_docs_path)

    # 3. Copy to project root for easy user discovery
    pdf_root_path = "LightGCN_Project_Report.pdf"
    shutil.copyfile(pdf_docs_path, pdf_root_path)
    print(f"Copied root PDF to {pdf_root_path}")

    print("\n[SUCCESS] HTML and PDF reports are ready!")

if __name__ == "__main__":
    main()
