# Phase 4 to Phase 14 Results Summary

Compact archive of historical model results before cleanup.

## Direction / Classification Progression

| Phase | Key Model | Main Direction Metric |
|---|---|---|
| 4 | Single-modality baselines | Price AUC 0.524, Document AUC 0.554, Surprise AUC 0.631 |
| 5 | Graph GAT | AUC 0.526 (No-GAT AUC 0.503) |
| 6 | First multimodal fusion | Direction AUC 0.516 |
| 7 | Ablations | Fusion AUC 0.516, GAT delta +0.023 |
| 9 | Strategic upgrades | Best enhanced-features AUC 0.523 |
| 11 | V2.1 fusion | Direction AUC 0.5917 |
| 12 | Vol-primary pivot | Direction AUC 0.5675 |
| 13 | HAR-RV features | Direction AUC 0.5848 (vol-primary variant) |
| 14 | HAR-RV skip fusion | Direction AUC 0.5907 |

## Volatility Progression (R2)

| Stage | Vol R2 |
|---|---|
| V2 Baseline | 0.335 |
| Historical Average | 0.3483 |
| Phase 12 Multimodal | 0.7719 |
| Phase 13 Vol-primary | 0.8665 |
| Phase 14 HAR-RV Skip | 0.9212 |
| HAR-RV benchmark | 0.9469 |

## Phase 14 Reference Metrics

- Vol R2: 0.9212166667
- RMSE: 0.0563266091
- MAE: 0.0337997824
- QLIKE: -0.9942297339
- Direction AUC: 0.5906700660
- Direction Accuracy: 0.5744013020
- Gap to HAR-RV benchmark: 0.0257230233

## Source Files Used

- models/phase4_results.json
- models/phase5_results.json
- models/phase6_results.json
- models/phase7_ablation_results.json
- models/phase9_results.json
- models/phase11_v2_1_results.json
- models/phase12_benchmark_results.json
- models/phase13_benchmark_results.json
- models/phase14_training_results.json
- models/phase14_benchmark_results.json
