"""
Script to train model.
"""
import logging
import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
import bdrk
from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.analyzer import ModelTypes
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
from sklearn import metrics
from sklearn.model_selection import train_test_split

from utils.constants import FEATURE_COLS, TARGET_COL

FEATURES_DATA = os.path.join(
    os.getenv("TEMP_DATA_BUCKET", ""),
    os.getenv("FEATURES_DATA", "doc/data/features.csv"),
)
LR = float(os.getenv("LR", 0.05))
NUM_LEAVES = int(os.getenv("NUM_LEAVES", 10))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
OUTPUT_MODEL_PATH = os.getenv("OUTPUT_MODEL_NAME", "doc/data/lgb_model.pkl")
OUTPUT_HISTOGRAM = os.getenv("OUTPUT_HISTOGRAM", "doc/data/histogram.prom")


def compute_log_metrics(clf, x_val, y_val):
    """Compute and log metrics."""
    print("\tEvaluating using validation data")
    y_prob = clf.predict_proba(x_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)

    print(f"Accuracy          = {acc:.6f}")
    print(f"Precision         = {precision:.6f}")
    print(f"Recall            = {recall:.6f}")
    print(f"F1 score          = {f1_score:.6f}")
    print(f"ROC AUC           = {roc_auc:.6f}")
    print(f"Average precision = {avg_prc:.6f}")

    # Log metrics
    bdrk.log_metric("Accuracy", acc)
    bdrk.log_metric("Precision", precision)
    bdrk.log_metric("Recall", recall)
    bdrk.log_metric("F1 score", f1_score)
    bdrk.log_metric("ROC AUC", roc_auc)
    bdrk.log_metric("Avg precision", avg_prc)
    bdrk.log_binary_classifier_metrics(
        actual=y_val.astype(int).tolist(),
        probability=y_prob.flatten().tolist(),
    )

    # Only analyze for orchestrated run
    if os.getenv("BEDROCK_RUN_TRIGGER") == "bedrock":
        # Calculate and upload xafai metrics
        analyzer = ModelAnalyzer(clf, 'tree_model', model_type=ModelTypes.TREE).test_features(x_val)
        analyzer.test_labels(y_val.values).test_inference(y_pred)
        analyzer.analyze()


def main():
    """Train pipeline"""
    bdrk.init(project_id=os.environ.get("BEDROCK_PROJECT_ID", "churn-prediction-project"))

    with bdrk.start_run(
        pipeline_id=os.environ.get("BEDROCK_PIPELINE_ID", "new-churn-prediction"),
        environment_id=os.environ.get("BEDROCK_ENVIRONMENT_ID", "canary-dev"),
    ):
        model_data = pd.read_csv(FEATURES_DATA)

        print("\tSplitting train and validation data")
        x_train, x_val, y_train, y_val = train_test_split(
            model_data[FEATURE_COLS],
            model_data[TARGET_COL],
            test_size=0.2,
        )

        print("\tTrain model")
        clf = lgb.LGBMClassifier(
            num_leaves=NUM_LEAVES,
            learning_rate=LR,
            n_estimators=N_ESTIMATORS,
        )
        clf.fit(x_train, y_train)
        compute_log_metrics(clf, x_val, y_val)

        print("\tComputing metrics")
        selected = np.random.choice(model_data.shape[0], size=1000, replace=False)
        features = model_data[FEATURE_COLS].iloc[selected]
        inference = clf.predict_proba(features)[:, 1]

        ModelMonitoringService.export_text(
            features=features.iteritems(),
            inference=inference.tolist(),
            path=OUTPUT_HISTOGRAM
        )

        print("\tSaving model")
        with open(OUTPUT_MODEL_PATH, "wb") as model_file:
            pickle.dump(clf, model_file)
        bdrk.log_model(OUTPUT_MODEL_PATH)

if __name__ == "__main__":
    main()
