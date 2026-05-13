# trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

import joblib
import os
import numpy as np
import time

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest,
)

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN

from xgboost import XGBClassifier, XGBRegressor
from pyod.models.hbos import HBOS

from sklearn.base import BaseEstimator, TransformerMixin

# =========================================================
# MODEL CONFIG
# =========================================================

ALL_MODELS_CONFIG = {
    "Isolation Forest": {
        "class": IsolationForest,
        "params": {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
            },
            "contamination": {
                "type": "float",
                "default": 0.01,
                "min": 0.0,
                "max": 0.5,
            },
            "random_state": {
                "type": "int",
                "default": 42,
                "hidden": True,
            },
        },
    },
    "Local Outlier Factor": {
        "class": LocalOutlierFactor,
        "params": {
            "n_neighbors": {
                "type": "int",
                "default": 20,
                "min": 1,
                "max": 200,
            },
            "contamination": {
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 0.5,
            },
            "novelty": {
                "type": "bool",
                "default": True,
            },
        },
    },
    "One-Class SVM": {
        "class": OneClassSVM,
        "params": {
            "kernel": {
                "type": "str",
                "default": "rbf",
                "options": ["linear", "poly", "rbf", "sigmoid"],
            },
            "gamma": {
                "type": "str",
                "default": "scale",
                "options": ["scale", "auto"],
            },
            "nu": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
            },
        },
    },
    "HBOS": {
        "class": HBOS,
        "params": {
            "n_bins": {
                "type": "int",
                "default": 10,
                "min": 2,
                "max": 50,
            },
            "contamination": {
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 0.5,
            },
        },
    },
}


# =========================================================
# DATETIME
# =========================================================

COMMON_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
]


def detect_and_parse_datetime(series: pd.Series):

    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if series.dtype != object:
        return None

    try:
        parsed = pd.to_datetime(series, errors="coerce")
        success_rate = parsed.notna().mean()

        if success_rate >= 0.5:
            return parsed

    except:
        pass

    return None


def convert_datetime_columns(df: pd.DataFrame):

    df = df.copy()

    datetime_cols = []

    for col in df.columns:

        parsed = detect_and_parse_datetime(df[col])

        if parsed is not None:
            df[col] = parsed
            datetime_cols.append(col)

    return df, datetime_cols


def get_column_types(df: pd.DataFrame, datetime_cols: list):

    result = {}

    for col in df.columns:

        if col in datetime_cols:
            result[col] = "datetime"

        elif pd.api.types.is_numeric_dtype(df[col]):
            result[col] = "numeric"

        else:
            result[col] = "categorical"

    return result


# =========================================================
# DATE FEATURE
# =========================================================


class DateFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        result = pd.DataFrame(index=X.index)

        for col in X.columns:

            s = X[col]

            result[f"{col}_year"] = s.dt.year
            result[f"{col}_month"] = s.dt.month
            result[f"{col}_day"] = s.dt.day
            result[f"{col}_dayofweek"] = s.dt.dayofweek

        return result


# =========================================================
# TRAINER
# =========================================================


class MLTrainer:

    def __init__(
        self,
        df,
        target_column,
        task_type,
        selected_features,
        user_model_params=None,
        test_size=0.2,
    ):

        self.df, self.datetime_cols = convert_datetime_columns(df)

        self.target_column = target_column
        self.task_type = task_type
        self.selected_features = selected_features

        self.user_model_params = user_model_params if user_model_params else {}

        self.test_size = test_size

        self.results = {}

        self.all_trained_models = {}

        self.best_model_name = None

        self.preprocessor = None

        self.label_encoder = None

        self.model_evaluation_details = {}

        self.scored_datasets = {}

        self.training_durations = {}

    # =====================================================
    # PREPROCESSOR
    # =====================================================

    def _build_preprocessor(self, X):

        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        datetime_features = X.select_dtypes(include=["datetime64"]).columns.tolist()

        transformers = []

        if numeric_features:

            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="mean"),
                            ),
                            (
                                "scaler",
                                StandardScaler(),
                            ),
                        ]
                    ),
                    numeric_features,
                )
            )

        if categorical_features:

            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    categorical_features,
                )
            )

        if datetime_features:

            transformers.append(
                (
                    "date",
                    Pipeline(
                        [
                            (
                                "extractor",
                                DateFeatureExtractor(),
                            ),
                            (
                                "imputer",
                                SimpleImputer(strategy="mean"),
                            ),
                            (
                                "scaler",
                                StandardScaler(),
                            ),
                        ]
                    ),
                    datetime_features,
                )
            )

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    # =====================================================
    # OUTLIER SCORE
    # =====================================================

    def _get_outlier_scores(self, pipeline, X):

        model = pipeline.named_steps["classifier"]

        X_processed = pipeline.named_steps["preprocessor"].transform(X)

        if hasattr(model, "decision_function"):

            scores = model.decision_function(X_processed)

        elif hasattr(model, "score_samples"):

            scores = model.score_samples(X_processed)

        else:

            raise ValueError(f"{type(model).__name__} " f"does not support scoring")

        scores = np.array(scores).astype(float)

        # lower = more suspicious
        if np.mean(scores) > 0:
            scores = -scores

        return scores

    # =====================================================
    # INSIGHTS
    # =====================================================

    def _generate_outlier_insights(self, df):

        insights = []

        if "Gross_Transaction_Amount" in df.columns:

            top_ratio = df.nsmallest(20, "outlier_score")[
                "Gross_Transaction_Amount"
            ].mean() / max(df["Gross_Transaction_Amount"].mean(), 1)

            if top_ratio > 3:

                insights.append("Outlier didominasi transaksi besar.")

        if "txn_vs_personal_avg" in df.columns:

            deviation_ratio = (df["txn_vs_personal_avg"] > 3).mean()

            if deviation_ratio > 0.3:

                insights.append("Behavioral deviation cukup signifikan.")

        if "Risk_Profile" in df.columns and "Fund_Risk_Level" in df.columns:

            mismatch = (df["Risk_Profile"] != df["Fund_Risk_Level"]).mean()

            if mismatch > 0.3:

                insights.append("Banyak risk mismatch terdeteksi.")

        if not insights:

            insights.append("Distribusi outlier terlihat cukup merata.")

        return insights

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    def _generate_recommendations(self, outlier_percentage, insights):

        recs = []

        if outlier_percentage > 20:

            recs.append("Coba contamination lebih kecil.")

        if any("transaksi besar" in x for x in insights):

            recs.append("Model terlihat terlalu amount-centric.")

            recs.append("Coba feature engineering behavioral features.")

        if outlier_percentage < 2:

            recs.append("Coba model lain dengan sensitivitas lebih tinggi.")

        if not recs:

            recs.append("Model terlihat cukup stabil untuk analisa lanjutan.")

        return recs

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, selected_models):

        X = self.df[self.selected_features]

        self._build_preprocessor(X)

        self._train_anomaly(X, selected_models)

        return (self.results, self.all_trained_models)

    # =====================================================
    # ANOMALY TRAINING
    # =====================================================

    def _train_anomaly(self, X, selected_models):

        print("========== ANOMALY TRAINING ==========")

        # =====================================================
        # FEATURE STATS FOR DYNAMIC EXPLAINABILITY
        # =====================================================

        self.feature_stats = {}

        numeric_cols = X.select_dtypes(include=["number"]).columns

        for col in numeric_cols:

            self.feature_stats[col] = {
                "mean": float(X[col].mean()),
                "std": float(X[col].std()),
            }

        for model_name in selected_models:

            try:

                config = ALL_MODELS_CONFIG[model_name]

                model_class = config["class"]

                default_params = {k: v["default"] for k, v in config["params"].items()}

                actual_params = {
                    **default_params,
                    **self.user_model_params.get(model_name, {}),
                }

                model = model_class(**actual_params)

                pipeline = Pipeline(
                    [
                        (
                            "preprocessor",
                            self.preprocessor,
                        ),
                        (
                            "classifier",
                            model,
                        ),
                    ]
                )

                start_time = time.time()

                pipeline.fit(X)

                duration = round(time.time() - start_time, 2)

                preds = pipeline.predict(X)

                scores = self._get_outlier_scores(pipeline, X)

                self.score_stats = {
                    "p0_1": float(np.percentile(scores, 0.1)),
                    "p1": float(np.percentile(scores, 1)),
                    "p3": float(np.percentile(scores, 3)),
                    "mean": float(np.mean(scores)),
                }

                results_df = X.copy()

                results_df["outlier_score"] = scores

                results_df["outlier_label"] = preds

                results_df["outlier_result"] = results_df["outlier_label"].map(
                    {
                        -1: "outlier",
                        1: "normal",
                    }
                )

                # =====================================================
                # SEVERITY LEVEL
                # =====================================================

                severity_levels = []

                for label, score in zip(
                    results_df["outlier_label"],
                    results_df["outlier_score"],
                ):

                    if label == 1:

                        severity = "normal"

                    else:

                        if score < -0.22:

                            severity = "critical"

                        elif score < -0.18:

                            severity = "high"

                        elif score < -0.12:

                            severity = "medium"

                        else:

                            severity = "low"

                    severity_levels.append(severity)

                results_df["severity_level"] = severity_levels

                # =====================================================
                # ABSOLUTE SCORE
                # =====================================================

                results_df["anomaly_score_abs"] = results_df["outlier_score"].abs()

                # =====================================================
                # DYNAMIC EXPLAINABILITY
                # =====================================================

                reasons_all = []

                for _, row in results_df.iterrows():

                    row_reasons = []

                    for feature, value in row.items():

                        if feature not in self.feature_stats:
                            continue

                        if not isinstance(
                            value,
                            (int, float, np.integer, np.floating),
                        ):
                            continue

                        stats = self.feature_stats[feature]

                        mean = stats["mean"]
                        std = stats["std"]

                        if std == 0:
                            continue

                        zscore = abs((value - mean) / std)

                        if zscore > 3:

                            row_reasons.append(f"{feature} extremely abnormal")

                        elif zscore > 2:

                            row_reasons.append(f"{feature} unusually deviated")

                    if not row_reasons:

                        if row["outlier_label"] == -1:

                            row_reasons.append("General behavioral anomaly")

                        else:

                            row_reasons.append("Behavior within normal range")

                    reasons_all.append(" | ".join(row_reasons))

                results_df["why_flagged"] = reasons_all

                # =====================================================
                # SORTING
                # =====================================================

                results_df = results_df.sort_values(
                    by="outlier_score",
                    ascending=True,
                ).reset_index(drop=True)

                results_df["outlier_rank"] = results_df.index + 1

                total_outliers = int((results_df["outlier_label"] == -1).sum())

                outlier_pct = round(
                    (total_outliers / len(results_df)) * 100,
                    2,
                )

                insights = self._generate_outlier_insights(results_df)

                recommendations = self._generate_recommendations(
                    outlier_pct,
                    insights,
                )

                summary = {
                    "total_rows": len(results_df),
                    "total_outliers_detected": total_outliers,
                    "outlier_percentage": outlier_pct,
                    "lowest_outlier_score": float(results_df["outlier_score"].min()),
                    "highest_outlier_score": float(results_df["outlier_score"].max()),
                    "average_outlier_score": float(results_df["outlier_score"].mean()),
                    "training_duration": duration,
                    "model_name": model_name,
                    "model_configuration": actual_params,
                    "insights": insights,
                    "recommendations": recommendations,
                }

                self.results[model_name] = summary

                self.model_evaluation_details[model_name] = {
                    "summary": summary,
                    "scored_dataset": results_df,
                }

                self.scored_datasets[model_name] = results_df

                self.training_durations[model_name] = duration

                self.all_trained_models[model_name] = pipeline

                print(f"SUCCESS: {model_name}")

            except Exception as e:

                print(f"ERROR TRAINING " f"{model_name}: {e}")

                self.results[model_name] = {"error": str(e)}

        self._select_best_model()

    # =====================================================
    # BEST MODEL
    # =====================================================

    def _select_best_model(self):

        best_model = None

        best_score = np.inf

        for model_name, metrics in self.results.items():

            if "error" in metrics:
                continue

            score = metrics.get("average_outlier_score", np.inf)

            if score < best_score:

                best_score = score
                best_model = model_name

        self.best_model_name = best_model

        print(f"BEST MODEL: " f"{self.best_model_name}")

    # =====================================================
    # SAVE
    # =====================================================

    def save_best_model(self, project_name):

        if (
            not self.best_model_name
            or self.best_model_name not in self.all_trained_models
        ):
            return None

        artifacts_dir = os.path.join(
            "projects",
            project_name,
            "artifacts",
        )

        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(artifacts_dir, f"{self.best_model_name}_model.joblib")

        joblib.dump(
            {
                "pipeline": self.all_trained_models[self.best_model_name],
                "label_encoder": self.label_encoder,
                "datetime_cols": self.datetime_cols,
                "evaluation_details": self.model_evaluation_details.get(
                    self.best_model_name
                ),
            },
            model_path,
        )

        return model_path


# =========================================================
# AVAILABLE MODELS
# =========================================================


def get_available_models(task_type):

    if task_type == "anomaly":

        return [
            "Isolation Forest",
            "Local Outlier Factor",
            "One-Class SVM",
            "HBOS",
        ]

    return []
