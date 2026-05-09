"""Train an unsupervised clustering pipeline and save run artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.eval.metrics import clustering_metrics
from src.features.preprocess import build_preprocessor
from src.models.clusterer import build_cluster_model
from src.utils.common import ensure_dir, load_yaml, set_seed


def _build_cluster_profile(df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
    """Create a compact per-cluster profile for teaching interpretation."""
    profile = pd.DataFrame(df[label_col].value_counts(dropna=False)).reset_index()
    profile.columns = [label_col, "size"]
    profile["share"] = profile["size"] / profile["size"].sum()

    numeric_cols = [
        col for col in df.select_dtypes(include=["number", "bool"]).columns
        if col != label_col
    ]
    if numeric_cols:
        numeric_summary = (
            df.groupby(label_col, dropna=False)[numeric_cols]
            .median(numeric_only=True)
            .add_prefix("median_")
            .reset_index()
        )
        profile = profile.merge(numeric_summary, on=label_col, how="left")

    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and col != label_col
    ]
    for column in categorical_cols:
        mode_series = (
            df.groupby(label_col, dropna=False)[column]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
            .rename(f"mode_{column}")
            .reset_index()
        )
        profile = profile.merge(mode_series, on=label_col, how="left")

    return profile.sort_values("size", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clustering pipeline")
    parser.add_argument("--config", default="configs/train/titanic_kmeans_unsupervised.yaml")
    args = parser.parse_args()

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(train_cfg["paths"]["data_config"])
    feature_cfg = load_yaml(train_cfg["paths"]["feature_config"])
    model_cfg = load_yaml(train_cfg["paths"]["model_config"])

    set_seed(int(train_cfg.get("seed", 42)))

    dataset_path = Path(data_cfg["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    target_col = data_cfg.get("target_column", "target")
    drop_cols = feature_cfg.get("drop_columns", [])
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")

    preprocessor = build_preprocessor(X, scale_numeric=bool(feature_cfg.get("scale_numeric", True)))
    cluster_model = build_cluster_model(
        model_type=model_cfg.get("model_type", "kmeans"),
        model_params=model_cfg.get("params", {}),
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", cluster_model)])
    labels = pipeline.fit_predict(X)
    X_processed = pipeline.named_steps["preprocessor"].transform(X)

    metrics = clustering_metrics(X_processed, labels)

    run_dir = ensure_dir(Path(train_cfg["paths"]["runs_dir"]) / train_cfg["run_name"])
    artifacts = train_cfg.get("artifacts", {})

    model_file = artifacts.get("model_file", "model.joblib")
    metrics_file = artifacts.get("metrics_file", "metrics.json")
    params_file = artifacts.get("params_file", "params.json")
    assignments_file = artifacts.get("assignments_file", "cluster_assignments.csv")
    profile_file = artifacts.get("profile_file", "cluster_profile.csv")

    joblib.dump(pipeline, run_dir / model_file)
    (run_dir / metrics_file).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    params = {
        "train_config": args.config,
        "data_config": train_cfg["paths"]["data_config"],
        "feature_config": train_cfg["paths"]["feature_config"],
        "model_config": train_cfg["paths"]["model_config"],
    }
    (run_dir / params_file).write_text(json.dumps(params, indent=2), encoding="utf-8")

    cluster_df = X.copy()
    if target_col in df.columns:
        cluster_df[target_col] = df[target_col].values
    cluster_df["cluster"] = labels
    cluster_df.to_csv(run_dir / assignments_file, index=False)

    profile_df = _build_cluster_profile(cluster_df, label_col="cluster")
    profile_df.to_csv(run_dir / profile_file, index=False)

    bundle_info = {
        "task": "unsupervised_clustering",
        "model_type": model_cfg.get("model_type", "unknown"),
        "run_name": train_cfg["run_name"],
        "model_file": model_file,
        "features": X.columns.tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_config": args.config,
    }
    (run_dir / "bundle_info.json").write_text(json.dumps(bundle_info, indent=2), encoding="utf-8")

    summary_path = Path(train_cfg["paths"]["runs_dir"]) / "summary.csv"
    summary_row = {
        "run_name": train_cfg["run_name"],
        "model": model_cfg.get("model_type", "unknown"),
        "calibrated": "",
        "accuracy": "",
        "f1": "",
        "roc_auc": "",
        "cv_score_mean": "",
        "cv_scoring": "",
        "cluster_count": metrics.get("cluster_count", ""),
        "noise_ratio": metrics.get("noise_ratio", ""),
        "silhouette": metrics.get("silhouette", ""),
        "davies_bouldin": metrics.get("davies_bouldin", ""),
        "calinski_harabasz": metrics.get("calinski_harabasz", ""),
        "notes": "unsupervised_clustering",
    }

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved unsupervised run artifacts to: {run_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
