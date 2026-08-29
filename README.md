# Explainable Edge-Aware Graph Attention & BiLSTM for Spatiotemporal Crash Risk Forecasting in Mixed Traffic

**An Empirical Study on Dhaka Traffic Video**

This repository contains the full pipeline, code, and experiments for a thesis project that forecasts frame-level crash risk (**Safe / Caution / Risk**) in Dhaka's mixed, non-lane-based traffic using ordinary smartphone video. Instead of relying on sparse, unreliable crash records, the pipeline derives dense, physically-grounded risk labels from surrogate safety measures (SSMs) — time-to-collision (TTC), distance at closest point of approach (DCPA), and closing speed — and uses them to train an **Edge-Aware Graph Attention Network (GAT) combined with a BiLSTM** that jointly models spatial vehicle-to-vehicle interactions and their temporal evolution.

The proposed model is benchmarked head-to-head against Logistic Regression, Random Forest, and a graph-free BiLSTM ablation, under an identical, leakage-free 70/15/15 split, with explainability provided via permutation importance and GAT attention heatmaps.

> 📄 Full write-up: [`An_Empirical_Study_on_Dhaka_Traffic_Video_Paper.pdf`](An_Empirical_Study_on_Dhaka_Traffic_Video_Paper.pdf)

---

## Why This Project

Road traffic injuries are one of the leading causes of death worldwide, with the burden falling disproportionately on low- and middle-income countries with heterogeneous, non-lane-based traffic. Dhaka is a representative case: rickshaws, buses, private cars, motorcycles, and pedestrians share the same carriageway with little lane discipline.

Two structural problems motivate this work:

1. **Sparse ground truth** — historical crash records in Bangladesh are sparse, inconsistently reported, and rarely geo-referenced, ruling out the large labeled crash datasets that deep models typically need.
2. **Reactive systems** — most existing crash-detection systems flag a collision *after* it happens rather than forecasting risk far enough ahead to allow a warning or intervention.

This project takes a **proactive, video-based alternative**: derive dense risk labels directly from everyday traffic video using surrogate safety measures, and train a model that forecasts near-miss risk several seconds ahead.

---

## Pipeline Overview

```
Smartphone traffic video (overbridge-recorded)
        │
        ▼
YOLOv8n Object Detection
        │
        ▼
ByteTrack Multi-Object Tracking
        │
        ▼
Trajectory Extraction (position, velocity, acceleration)
        │
        ▼
Pairwise Surrogate Safety Measures (TCA, TTC, DCPA, closing speed)
        │
        ▼
Automatic Risk Labeling (Safe / Caution / Risk)
        │
        ▼
Dynamic Per-Frame Traffic Graph Construction (nodes = road users, edges = interactions)
        │
        ▼
Edge-Aware Graph Attention Network (spatial interaction learning)
        │
        ▼
BiLSTM (temporal risk evolution over a 10-frame window)
        │
        ▼
Risk Classification: Safe | Caution | Risk
        │
        ▼
Explainability: GAT Attention Heatmaps + Permutation Feature Importance
```

---

## Dataset

All trajectory, interaction, and risk-label data come from **five traffic videos** recorded on ordinary smartphones from pedestrian overbridges at different locations in and around Dhaka (not fixed CCTV).

| Video | Processed frames | Tracked detections |
|---|---|---|
| Airport | 2,417 | 33,712 |
| Rajlokkhi | 1,219 | 22,207 |
| Ajampur1 | 1,208 | 20,344 |
| Dhaka test clip | 102 | 1,084 |
| Arambag (double lane) | 203 | 245 |
| **Total** | **5,149** | **77,592** |

A separate dataset — the **Bangladeshi Traffic Flow Dataset** (Mendeley Data, via Kaggle mirror; 47,356 raw traffic images) — was used *only* to sanity-check a pretrained YOLOv8n detector on local traffic imagery. It was **not** used for fine-tuning and contributes no labels or trajectories.

**Resulting label distribution:**
- 513,035 pairwise interaction samples → 359,124 safe / 102,607 caution / 51,304 risk
- 5,071 labeled frames (7:2:1 imbalance) → 3,549 safe / 1,014 caution / 508 risk
- 5,026 overlapping 10-frame graph sequences (70/15/15 split: 3,518 train / 754 val / 754 test)

> ⚠️ Raw video files and large intermediate `.csv`/`.pt` artifacts are **not** included in this repository due to size. See [Reproducing the Pipeline](#reproducing-the-pipeline) below for how to regenerate them.

---

## Models

| Model | Description | Trainable params |
|---|---|---|
| **Logistic Regression** | Linear, class-balanced baseline on 27 hand-engineered frame-level aggregate features | — |
| **Random Forest** | 300 class-balanced trees, max depth 10 | — |
| **BiLSTM (sequence-only)** | Bidirectional LSTM on pooled per-frame node features, no graph/edge modeling — isolates the value of graph attention | 57,859 |
| **Edge-Aware GAT + BiLSTM (proposed)** | Two stacked edge-aware GAT layers per frame + BiLSTM over the resulting embedding sequence | 106,053 |

### Edge-Aware GAT Attention

For node features `h_i` and edge features `e_ij` (both projected to 64-dim), the attention score for pair `(i, j)` is:

```
α_ij = softmax_j( MLP([h_i ‖ h_j ‖ e_ij]) )
```

computed with a two-layer LeakyReLU MLP and softmax over valid neighbors (including self-loops), followed by masked mean pooling into a single 64-dim graph embedding per frame. The resulting `T=10`-step sequence feeds a single-layer BiLSTM followed by a two-layer classification head.

---

## Results (5-video dataset, test set)

| Rank | Model | Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC |
|---|---|---|---|---|---|
| 1 | **GAT + BiLSTM (proposed)** | **0.821** | **0.695** | **0.812** | 0.889 |
| 2 | Random Forest | 0.756 | 0.669 | 0.767 | **0.890** |
| 3 | Logistic Regression | 0.615 | 0.500 | 0.642 | 0.770 |
| 4 | BiLSTM (sequence only) | 0.606 | 0.439 | 0.620 | 0.696 |

**Key findings:**

- The proposed **GAT+BiLSTM is the strongest model overall**, narrowly ahead of Random Forest and clearly ahead of the linear and graph-free baselines.
- The controlled ablation against the BiLSTM-only baseline (identical temporal head, training procedure, and data — differing *only* in the presence of graph-attention layers) shows a **25.6-point absolute (58% relative) macro-F1 gain**, isolating the specific contribution of graph attention.
- On the safety-critical **risk** class, GAT+BiLSTM has the highest precision (78.7%, few false alarms) but lower recall (48.7%) than Random Forest (67.1%) — the two models have complementary strengths as an alarm system.
- **Dataset scale, not architecture, was the dominant factor.** On an earlier, smaller four-video/2,654-frame dataset, Random Forest led by a wide margin (macro-F1 0.748 vs. 0.458). Adding one additional, denser video — no architecture changes — reversed the ranking entirely, confirming the graph model was data-starved rather than poorly suited to the problem.
- Permutation importance and GAT attention heatmaps both identify **minimum time-to-collision** as the single most informative risk signal.

Full class-wise metrics, confusion matrices, ROC curves, and attention heatmaps are in the paper (Section V) and reproduced by the notebook.

---

## Repository Structure

```
.
├── README.md
├── paper/
│   └── An_Empirical_Study_on_Dhaka_Traffic_Video_Paper.pdf   # Full thesis write-up
├── notebooks/
│   └── FinalThesisCode2_0.ipynb                              # End-to-end pipeline (Kaggle-based)
├── requirements.txt
└── LICENSE
```

The notebook (`notebooks/FinalThesisCode2_0.ipynb`) contains the complete pipeline in sequence:

1. **Detection sanity-check** — pretrained YOLOv8n evaluated on the Bangladeshi Traffic Flow Dataset
2. **Video tracking** — YOLOv8n + ByteTrack applied to the five smartphone videos to produce object trajectories
3. **Motion feature engineering** — speed, acceleration, class filtering
4. **Pairwise SSM computation** — TCA, TTC, DCPA, closing speed, safety radius
5. **Automatic risk labeling** — interaction-level and frame-level Safe/Caution/Risk labels
6. **Classical ML baselines** — Logistic Regression and Random Forest on 27 hand-engineered aggregate features
7. **Dynamic graph construction** — per-frame graphs (up to 20 nodes) grouped into 10-frame sequences
8. **BiLSTM-only ablation** — sequence model without graph structure
9. **Edge-Aware GAT + BiLSTM** — the proposed model, training, and evaluation
10. **Explainability** — permutation feature importance (Random Forest) and GAT attention heatmaps

> The notebook was originally developed and run on **Kaggle** (paths reference `/kaggle/input` and `/kaggle/working`). See below for notes on adapting it to run locally.

---

## Getting Started

### Requirements

- Python 3.10+
- PyTorch
- Ultralytics YOLOv8 (`ultralytics`)
- OpenCV (`opencv-python`)
- scikit-learn
- pandas, numpy
- matplotlib

Install with:

```bash
pip install -r requirements.txt
```

A starting `requirements.txt`:

```
torch
ultralytics
opencv-python
scikit-learn
pandas
numpy
matplotlib
jupyter
```

### Reproducing the Pipeline

1. **Get the data.**
   - Detector sanity-check images: [Bangladeshi Traffic Flow Dataset](https://www.kaggle.com/datasets/ari7889/bangladesh-traffic) (Mendeley Data / Kaggle mirror).
   - The five self-recorded traffic videos used for trajectories and risk labels are not publicly redistributed in this repo; substitute your own overbridge-style traffic footage, or contact the author for access.
2. **Update paths.** The notebook was built on Kaggle and references `/kaggle/input/...` and `/kaggle/working/...`. Replace these with local paths (e.g. `data/raw/`, `data/processed/`) before running locally.
3. **Run the notebook top to bottom.** Each stage saves its intermediate output (`trajectory_clean_motion.csv`, `interaction_risk_labels.csv`, `frame_risk_labels.csv`, `graph_node_features.csv`, `graph_edge_features.csv`, `gat_bilstm_graph_sequence_dataset.pt`, etc.) so later stages can be re-run independently once earlier artifacts exist.
4. **Trained models** (`bilstm_baseline_model.pt`, `gat_bilstm_model.pt`) and result tables/plots are written out at the end of their respective sections.

---

## Explainability

- **Random Forest:** permutation feature importance (20 repeats, macro-F1 scoring) on the test set — `min_ttc_sec` is consistently the dominant feature.
- **GAT + BiLSTM:** raw attention matrix `α_ij` extracted from the final GAT layer and visualized as a source/target heatmap. At the larger data scale, attention becomes sharply concentrated on a few source-target pairs rather than diffuse, indicating the model is learning *which* pairwise interaction matters rather than relying on fixed aggregate statistics.

---

## Limitations

- Risk labels are generated automatically from surrogate safety measures and have **not been validated against real crash or near-miss ground truth** on these specific corridors; absolute performance numbers should be read as relative model comparisons, not calibrated real-world risk probabilities.
- The dataset comes from only **five videos at five physical locations**; the train/val/test split is stratified by label but not by video, so some scene-specific leakage across splits is possible.
- The automatic labeling rule and the classical baselines' hand-engineered features share the same underlying TTC/DCPA computation — by design, but it means the classical baselines benefit from feature-label alignment a deployed system without ground-truth SSMs would not have.
- All recordings were captured during **daytime, dry conditions, from a fixed overbridge vantage point**; performance at night, in rain/fog, or from a moving camera (e.g. dashcam) is untested.
- Results are from **single runs per model** rather than averages over multiple random seeds, so exact gap magnitudes (though not their direction) carry some run-to-run variance typical of small-data deep learning.

---

## Future Work

- Continue adding video from more locations, times of day, and weather conditions — dataset scale was the dominant factor governing model ranking in this study.
- Combine both models' complementary strengths in an ensemble or two-stage cascade (Random Forest as a high-recall filter, GAT+BiLSTM as a high-precision, explainable confirmatory stage).
- Further ablations on GAT layer/head count and auxiliary SSM regression to close the risk-class recall gap architecturally.
- Grad-CAM-style overlays on the source video, combined with GAT attention weights, to highlight which vehicles triggered an alert directly on the frame — moving toward a real-time, explainable road-safety tool for Dhaka's mixed-traffic corridors.

---

## Citation

If you use this work, please cite:

```bibtex
@techreport{dhaka_crash_risk_gat_bilstm,
  title        = {Explainable Edge-Aware Graph Attention and BiLSTM Framework for
                   Spatiotemporal Crash Risk Forecasting in Mixed Traffic:
                   An Empirical Study on Dhaka Traffic Video},
  author       = {[Your Name]},
  year         = {2026},
  note         = {Thesis / project report}
}
```

## References

Key works this project builds on:

- Jocher et al., *Ultralytics YOLOv8*, 2023.
- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
- Veličković et al., *Graph Attention Networks*, ICLR 2018.
- Minderhoud & Bovy, *Extended Time-to-Collision Measures for Road Traffic Safety Assessment*, Accident Analysis & Prevention, 2001.
- Nippani et al., *Graph Neural Networks for Road Safety Modeling*, NeurIPS Datasets and Benchmarks Track, 2023.

Full reference list in the [paper](paper/An_Empirical_Study_on_Dhaka_Traffic_Video_Paper.pdf).

