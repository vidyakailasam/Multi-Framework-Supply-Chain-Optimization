"""
=============================================================================
Availability Bridge - Complete ML Pipeline
=============================================================================
Data Source : Supply chain Bridge across India
Frameworks  : MLflow · Scikit-Learn · XGBoost · TensorFlow/Keras · PyTorch · PySpark MLlib
Tasks       : Clustering (unsupervised) + Classification (supervised)
=============================================================================

Install dependencies:
    pip install mlflow scikit-learn xgboost tensorflow torch pyspark pandas numpy matplotlib seaborn

Run:
    python availability_ml_pipeline.py
    mlflow ui          # Then open http://localhost:5000 to view experiment results
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.tensorflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow import keras
import os
import json

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY  (works on Windows, Mac, and Linux)
# Creates an  "ml_outputs"  folder next to this script
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    """Return full path for an output file."""
    return os.path.join(OUTPUT_DIR, filename)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RAW DATA 
# ─────────────────────────────────────────────────────────────────────────────
RAW = {
    "node": ["TVN","HUB","VSK","BBN","GUW","IDR","JAI","LUD","NAG","PAT"],
    "demand_selection":       [148403,93265,186684,219762,234316,188707,223838,179382,222515,186694],
    "total_cardinality_OH":   [57251, 59515,59995, 117176,226238,113998,210997,201814,122240,188407],
    "demand_contr_avail":     [27994, 35103,36666, 54559, 65856, 46293, 57405, 59308, 50461, 55928],
    "no_demand_avail":        [29257, 24412,23329, 62617, 160382,67705, 153592,142506,71779, 132479],
    "gap_pct":                [81.1,  62.4,  80.4,  75.2,  71.9,  75.5,  74.4,  66.9,  77.3,  70.0],
    "network_doc_lt7_pct":    [20.1,  24.1,  21.9,  23.3,  24.6,  23.7,  24.9,  25.6,  23.3,  24.8],
    "network_doc_gt7_inv_lt7":[0.2,   0.1,   1.3,   2.4,   2.4,   1.7,   2.2,   1.5,   2.0,   2.0],
    "feeder_instock_loss_pct":[9.1,   6.1,   7.8,   7.6,   8.1,   6.9,   12.2,  6.8,   7.2,   9.2],
    "mix_change_pct":         [6.3,   4.9,   19.4,  24.2,  14.4,  19.2,  16.9,  8.0,   18.1,  13.1],
    "non_sort_loss_pct":      [6.8,   7.3,   7.5,   6.5,   7.4,   5.5,   6.1,   5.5,   6.3,   6.9],
    "s3p_registration_pct":   [31.5,  18.6,  32.1,  22.9,  19.5,  19.2,  16.5,  13.0,  13.5,  14.9],
}

df_raw = pd.DataFrame(RAW)
print("=" * 70)
print("RAW DATA (10 nodes × 12 features)")
print("=" * 70)
print(df_raw.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
df = df_raw.copy()

# Derived ratio features
df["avail_rate"]           = df["demand_contr_avail"] / df["total_cardinality_OH"]
df["demand_coverage"]      = df["demand_contr_avail"] / df["demand_selection"]
df["no_demand_ratio"]      = df["no_demand_avail"]    / df["total_cardinality_OH"]
df["total_loss_pct"]       = (df["network_doc_lt7_pct"] +
                               df["feeder_instock_loss_pct"] +
                               df["mix_change_pct"] +
                               df["non_sort_loss_pct"] +
                               df["s3p_registration_pct"])

# Classification label: HIGH / MEDIUM / LOW availability gap risk
# Using gap_pct thresholds: >78 → HIGH, 70-78 → MEDIUM, <70 → LOW
df["risk_label"] = pd.cut(
    df["gap_pct"],
    bins=[0, 70, 78, 100],
    labels=["LOW", "MEDIUM", "HIGH"]
).astype(str)

le = LabelEncoder()
df["risk_label_enc"] = le.fit_transform(df["risk_label"])

FEATURE_COLS = [
    "avail_rate", "demand_coverage", "no_demand_ratio",
    "network_doc_lt7_pct", "network_doc_gt7_inv_lt7",
    "feeder_instock_loss_pct", "mix_change_pct",
    "non_sort_loss_pct", "s3p_registration_pct",
    "total_loss_pct"
]

X = df[FEATURE_COLS].values
y = df["risk_label_enc"].values
nodes = df["node"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeature matrix shape : {X_scaled.shape}")
print(f"Label distribution   : {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Classes              : {le.classes_}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SYNTHETIC DATA AUGMENTATION  (10 rows → 500 rows for model training)
#     Real project: replace with actual historical weekly snapshots
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
N_SYNTH = 500

X_synth = np.vstack([
    X_scaled[np.random.choice(len(X_scaled), N_SYNTH, replace=True)] +
    np.random.normal(0, 0.15, (N_SYNTH, X_scaled.shape[1]))
])
y_synth = y[np.random.choice(len(y), N_SYNTH, replace=True)]

X_train, X_test, y_train, y_test = train_test_split(
    X_synth, y_synth, test_size=0.2, random_state=42, stratify=y_synth
)

print(f"\nSynthetic dataset     : {N_SYNTH} samples (augmented from 10 real nodes)")
print(f"Train / Test split   : {len(X_train)} / {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MLFLOW EXPERIMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT = "Availability_Bridge_WK05"
mlflow.set_experiment(EXPERIMENT)
print(f"\nMLflow experiment    : '{EXPERIMENT}'")
print("Run: mlflow ui  →  http://localhost:5000\n")

# ═════════════════════════════════════════════════════════════════════════════
# ████  SECTION A : CLUSTERING  (Unsupervised)  ████
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION A: CLUSTERING (Unsupervised)")
print("=" * 70)

# PCA for 2-D visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

clustering_results = {}

# ── A1. K-Means ──────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="KMeans_Clustering"):
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("task", "clustering")

    k = 3
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    sil  = silhouette_score(X_scaled, km_labels)
    db   = davies_bouldin_score(X_scaled, km_labels)

    mlflow.log_param("n_clusters", k)
    mlflow.log_metric("silhouette_score", sil)
    mlflow.log_metric("davies_bouldin_score", db)
    mlflow.sklearn.log_model(km, "kmeans_model")

    clustering_results["KMeans"] = {"labels": km_labels, "sil": sil, "db": db}
    print(f"[KMeans]    k={k}  Silhouette={sil:.3f}  Davies-Bouldin={db:.3f}")
    print(f"  Node assignments: { {n: int(l) for n, l in zip(nodes, km_labels)} }")

# ── A2. DBSCAN ───────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="DBSCAN_Clustering"):
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("task", "clustering")

    db_model = DBSCAN(eps=1.5, min_samples=2)
    db_labels = db_model.fit_predict(X_scaled)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    mlflow.log_param("eps", 1.5)
    mlflow.log_param("min_samples", 2)
    mlflow.log_metric("n_clusters_found", n_clusters_db)

    clustering_results["DBSCAN"] = {"labels": db_labels, "n_clusters": n_clusters_db}
    print(f"[DBSCAN]    Clusters found={n_clusters_db}")
    print(f"  Node assignments: { {n: int(l) for n, l in zip(nodes, db_labels)} }")

# ── A3. Agglomerative ────────────────────────────────────────────────────────
with mlflow.start_run(run_name="Agglomerative_Clustering"):
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("task", "clustering")

    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)

    sil_agg = silhouette_score(X_scaled, agg_labels)
    db_agg  = davies_bouldin_score(X_scaled, agg_labels)

    mlflow.log_param("n_clusters", 3)
    mlflow.log_param("linkage", "ward")
    mlflow.log_metric("silhouette_score", sil_agg)
    mlflow.log_metric("davies_bouldin_score", db_agg)

    clustering_results["Agglomerative"] = {"labels": agg_labels, "sil": sil_agg, "db": db_agg}
    print(f"[Agglomerative] k=3  Silhouette={sil_agg:.3f}  Davies-Bouldin={db_agg:.3f}")
    print(f"  Node assignments: { {n: int(l) for n, l in zip(nodes, agg_labels)} }")

# ── A4. Isolation Forest (Anomaly Detection) ─────────────────────────────────
with mlflow.start_run(run_name="IsolationForest_Anomaly"):
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("task", "anomaly_detection")

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_labels = iso.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal

    n_anomalies = (iso_labels == -1).sum()
    anomaly_nodes = nodes[iso_labels == -1].tolist()

    mlflow.log_param("contamination", 0.1)
    mlflow.log_metric("n_anomalies", n_anomalies)
    mlflow.log_param("anomaly_nodes", str(anomaly_nodes))

    print(f"[IsolationForest] Anomalies detected={n_anomalies}: {anomaly_nodes}")

# Visualise clusters
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Availability Bridge WK05 – Node Clustering", fontsize=14, fontweight="bold")

for ax, (method, res) in zip(axes, [
    ("KMeans", clustering_results["KMeans"]["labels"]),
    ("DBSCAN", clustering_results["DBSCAN"]["labels"]),
    ("Agglomerative", clustering_results["Agglomerative"]["labels"]),
]):
    labels = res if isinstance(res, np.ndarray) else res
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Set1", s=200, edgecolors="k")
    for i, node in enumerate(nodes):
        ax.annotate(node, (X_pca[i, 0] + 0.05, X_pca[i, 1] + 0.05), fontsize=8)
    ax.set_title(f"{method}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.tight_layout()
plt.savefig(out("clustering_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nClustering plot saved → clustering_plot.png")

# ═════════════════════════════════════════════════════════════════════════════
# ████  SECTION B : CLASSIFICATION  ████
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION B: CLASSIFICATION (Supervised) – Predict Availability Risk")
print("Classes: LOW / MEDIUM / HIGH gap risk")
print("=" * 70)

clf_metrics = {}

# ─────────────────────────────────────────────────────────────────────────────
# B1. Scikit-Learn: Random Forest
# ─────────────────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="RandomForest_sklearn"):
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("task", "classification")

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    mlflow.log_params({"n_estimators": 200, "max_depth": 5})
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.sklearn.log_model(rf, "random_forest_model")

    clf_metrics["RandomForest"] = acc_rf
    print(f"[RandomForest]  Accuracy = {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf,
          target_names=le.classes_, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# B2. Scikit-Learn: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="LogisticRegression_sklearn"):
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("task", "classification")

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    mlflow.log_params({"max_iter": 1000, "C": 1.0})
    mlflow.log_metric("accuracy", acc_lr)
    mlflow.sklearn.log_model(lr, "logistic_regression_model")

    clf_metrics["LogisticRegression"] = acc_lr
    print(f"[LogisticRegression]  Accuracy = {acc_lr:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# B3. XGBoost
# ─────────────────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="XGBoost_Classifier"):
    mlflow.set_tag("framework", "xgboost")
    mlflow.set_tag("task", "classification")

    params_xgb = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "num_class": len(le.classes_)
    }

    xgb_model = xgb.XGBClassifier(**{k: v for k, v in params_xgb.items()
                                      if k not in ("num_class",)})
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)

    mlflow.log_params(params_xgb)
    mlflow.log_metric("accuracy", acc_xgb)
    mlflow.xgboost.log_model(xgb_model, "xgboost_model")

    clf_metrics["XGBoost"] = acc_xgb
    print(f"[XGBoost]  Accuracy = {acc_xgb:.4f}")
    print(classification_report(y_test, y_pred_xgb,
          target_names=le.classes_, zero_division=0))

    # Feature importance
    importances = xgb_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": FEATURE_COLS, "importance": importances
    }).sort_values("importance", ascending=False)
    print("XGBoost Top Features:\n", feat_imp_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# B4. TensorFlow / Keras – Deep Neural Network
# ─────────────────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="TensorFlow_DNN"):
    mlflow.set_tag("framework", "tensorflow")
    mlflow.set_tag("task", "classification")

    n_classes = len(le.classes_)
    tf.random.set_seed(42)

    tf_model = keras.Sequential([
        keras.layers.InputLayer(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ], name="AvailabilityDNN")

    tf_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = tf_model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    _, acc_tf = tf_model.evaluate(X_test, y_test, verbose=0)
    y_pred_tf = np.argmax(tf_model.predict(X_test, verbose=0), axis=1)

    mlflow.log_params({
        "layers": "64-32-16",
        "dropout": 0.3,
        "optimizer": "adam",
        "lr": 0.001,
        "epochs_run": len(history.history["loss"])
    })
    mlflow.log_metric("accuracy", acc_tf)
    mlflow.tensorflow.log_model(tf_model, "tensorflow_dnn_model")

    clf_metrics["TensorFlow_DNN"] = acc_tf
    print(f"\n[TensorFlow DNN]  Accuracy = {acc_tf:.4f}")
    print(f"  Epochs trained   : {len(history.history['loss'])}")
    print(classification_report(y_pred_tf, y_test,
          target_names=le.classes_, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# B5. PyTorch – Neural Network
# ─────────────────────────────────────────────────────────────────────────────
class AvailabilityNet(nn.Module):
    """Fully-connected classifier for availability gap risk."""
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        return self.net(x)


with mlflow.start_run(run_name="PyTorch_Classifier"):
    mlflow.set_tag("framework", "pytorch")
    mlflow.set_tag("task", "classification")

    torch.manual_seed(42)
    n_classes_pt = len(le.classes_)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_te_t = torch.tensor(y_test,  dtype=torch.long)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    pt_model = AvailabilityNet(X_train.shape[1], n_classes_pt)
    criterion = nn.CrossEntropyLoss()
    optimizer_pt = optim.Adam(pt_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_pt, step_size=20, gamma=0.5)

    EPOCHS = 100
    pt_losses = []
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        pt_model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            optimizer_pt.zero_grad()
            loss = criterion(pt_model(xb), yb)
            loss.backward()
            optimizer_pt.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_dl)
        pt_losses.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss

    pt_model.eval()
    with torch.no_grad():
        logits = pt_model(X_te_t)
        y_pred_pt = torch.argmax(logits, dim=1).numpy()

    acc_pt = accuracy_score(y_test, y_pred_pt)

    mlflow.log_params({
        "epochs": EPOCHS,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 1e-4,
        "scheduler": "StepLR"
    })
    mlflow.log_metric("accuracy", acc_pt)
    mlflow.log_metric("final_train_loss", best_loss)
    mlflow.pytorch.log_model(pt_model, "pytorch_model")

    clf_metrics["PyTorch_NN"] = acc_pt
    print(f"\n[PyTorch NN]  Accuracy = {acc_pt:.4f}  |  Best Train Loss = {best_loss:.4f}")
    print(classification_report(y_test, y_pred_pt,
          target_names=le.classes_, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# B6. PySpark MLlib (GBT Classifier)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[PySpark MLlib] Initialising local Spark session …")
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import GBTClassifier, RandomForestClassifier as SparkRF
    from pyspark.ml.feature import VectorAssembler, StringIndexer
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    spark = SparkSession.builder \
        .appName("AvailabilityBridge_MLlib") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Rebuild DataFrame from real 10-node data for Spark
    spark_df = spark.createDataFrame(df[FEATURE_COLS + ["risk_label"]])
    idx = StringIndexer(inputCol="risk_label", outputCol="label")
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    gbt = GBTClassifier(maxIter=20, maxDepth=4, seed=42,
                        labelCol="label", featuresCol="features")
    pipeline = Pipeline(stages=[idx, assembler, gbt])

    train_sp, test_sp = spark_df.randomSplit([0.8, 0.2], seed=42)
    if test_sp.count() == 0:
        test_sp = spark_df

    with mlflow.start_run(run_name="PySpark_MLlib_GBT"):
        mlflow.set_tag("framework", "pyspark_mllib")
        mlflow.set_tag("task", "classification")
        mlflow.log_params({"maxIter": 20, "maxDepth": 4})

        model_spark = pipeline.fit(spark_df)
        predictions  = model_spark.transform(spark_df)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        acc_spark = evaluator.evaluate(predictions)
        mlflow.log_metric("accuracy", acc_spark)
        clf_metrics["PySpark_MLlib_GBT"] = acc_spark

    print(f"[PySpark MLlib GBT]  Accuracy = {acc_spark:.4f}")
    spark.stop()

except ImportError:
    print("[PySpark MLlib] PySpark not installed — skipping.")
    print("  Install: pip install pyspark")
except Exception as e:
    print(f"[PySpark MLlib] Error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# ████  SECTION C : RESULTS SUMMARY + VISUALISATIONS  ████
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION C: MODEL COMPARISON SUMMARY")
print("=" * 70)

summary_df = pd.DataFrame(
    list(clf_metrics.items()), columns=["Model", "Accuracy"]
).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print(summary_df.to_string(index=False))
best_model = summary_df.iloc[0]["Model"]
print(f"\n🏆  Best classifier: {best_model} ({summary_df.iloc[0]['Accuracy']:.4f})")

# ── Plot 1: Model accuracy comparison ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2ecc71" if v == summary_df["Accuracy"].max() else "#3498db"
          for v in summary_df["Accuracy"]]
bars = ax.barh(summary_df["Model"], summary_df["Accuracy"], color=colors, edgecolor="k")
ax.set_xlabel("Accuracy")
ax.set_title("Model Accuracy Comparison – Availability Gap Risk Classification")
ax.set_xlim(0, 1.1)
for bar, val in zip(bars, summary_df["Accuracy"]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(out("model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 2: XGBoost Feature Importance ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
feat_imp_df_sorted = feat_imp_df.sort_values("importance")
ax.barh(feat_imp_df_sorted["feature"], feat_imp_df_sorted["importance"],
        color="#e74c3c", edgecolor="k")
ax.set_title("XGBoost Feature Importance – Availability Gap Drivers")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(out("feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 3: Correlation heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr_df = df[FEATURE_COLS + ["gap_pct"]].corr()
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(out("correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 4: Gap % per node ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors_gap = ["#e74c3c" if g > 78 else "#f39c12" if g > 70 else "#2ecc71"
              for g in df["gap_pct"]]
ax.bar(df["node"], df["gap_pct"], color=colors_gap, edgecolor="k")
ax.axhline(78, color="red", linestyle="--", label="HIGH threshold (78%)")
ax.axhline(70, color="orange", linestyle="--", label="MEDIUM threshold (70%)")
ax.set_ylabel("Gap %")
ax.set_title("Availability Gap % by Fulfilment Node (WK05)")
ax.legend()
plt.tight_layout()
plt.savefig(out("gap_by_node.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 5: TF training curve ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"], label="train loss")
ax.plot(history.history["val_loss"], label="val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("TensorFlow DNN Training Curve")
ax.legend()
plt.tight_layout()
plt.savefig(out("tf_training_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 6: PyTorch loss curve ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(pt_losses, color="#9b59b6")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("PyTorch NN Training Loss Curve")
plt.tight_layout()
plt.savefig(out("pytorch_training_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll plots saved to {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS JSON
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "experiment": EXPERIMENT,
    "classification_accuracy": clf_metrics,
    "best_model": best_model,
    "clustering": {
        "KMeans_silhouette": clustering_results["KMeans"]["sil"],
        "Agglomerative_silhouette": clustering_results["Agglomerative"]["sil"],
        "DBSCAN_n_clusters": int(clustering_results["DBSCAN"]["n_clusters"]),
    },
    "node_risk_labels": dict(zip(nodes.tolist(), df["risk_label"].tolist())),
    "kmeans_node_clusters": {n: int(l) for n, l in zip(nodes, km_labels)},
}

with open(out("ml_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Results JSON saved → {out('ml_results.json')}")
print(f" All plots saved  → {OUTPUT_DIR}")
print("  Run  `mlflow ui`  to explore all experiment runs.")
print("=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
