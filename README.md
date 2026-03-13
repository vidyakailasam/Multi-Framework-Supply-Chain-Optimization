## Overview
End-to-end machine learning pipeline built on supply chain data which
Covers **clustering** (unsupervised) and **classification** (supervised) across six frameworks,
all tracked with **MLflow**.

---

## Data (10 Fulfilment Nodes)
| Node | Gap % | Risk Label |
|------|-------|------------|
| TVN  | 81.1% | HIGH       |
| VSK  | 80.4% | HIGH       |
| NAG  | 77.3% | MEDIUM     |
| IDR  | 75.5% | MEDIUM     |
| BBN  | 75.2% | MEDIUM     |
| JAI  | 74.4% | MEDIUM     |
| GUW  | 71.9% | MEDIUM     |
| PAT  | 70.0% | MEDIUM     |
| HUB  | 62.4% | LOW        |
| LUD  | 66.9% | LOW        |

**Engineered features:** availability rate, demand coverage, no-demand ratio,
total loss %, and all raw percentage drivers.

---

## Frameworks & Models

### Clustering (Unsupervised)
| Algorithm | Framework | Metric |
|-----------|-----------|--------|
| K-Means (k=3) | scikit-learn | Silhouette score |
| DBSCAN | scikit-learn | Clusters found |
| Agglomerative (Ward) | scikit-learn | Silhouette score |
| Isolation Forest | scikit-learn | Anomaly detection |

### Classification (Supervised) – Predict HIGH / MEDIUM / LOW gap risk
| Model | Framework | Logged With |
|-------|-----------|-------------|
| Random Forest | scikit-learn | mlflow.sklearn |
| Logistic Regression | scikit-learn | mlflow.sklearn |
| XGBoost Classifier | XGBoost | mlflow.xgboost |
| Deep Neural Network | TensorFlow / Keras | mlflow.tensorflow |
| Neural Network | PyTorch | mlflow.pytorch |
| GBT Classifier | PySpark MLlib | mlflow (custom) |

---

## Installation
```bash
pip install mlflow scikit-learn xgboost tensorflow torch pyspark \
            pandas numpy matplotlib seaborn
```

## Run
```bash
python availability_ml_pipeline.py

# Then view all runs in the MLflow UI:
mlflow ui
# Open http://localhost:5000
```

## Outputs
| File | Description |
|------|-------------|
| `clustering_plot.png` | PCA 2-D view of 3 clustering methods |
| `gap_by_node.png` | Gap % bar chart with risk thresholds |
| `feature_importance.png` | XGBoost top feature drivers |
| `correlation_heatmap.png` | Feature correlation matrix |
| `tf_training_curve.png` | TensorFlow loss/val-loss per epoch |
| `pytorch_training_curve.png` | PyTorch training loss curve |
| `model_comparison.png` | Accuracy comparison across all classifiers |
| `ml_results.json` | Full numeric results (JSON) |
| `mlruns/` | MLflow artefacts & metrics (auto-created) |

---
