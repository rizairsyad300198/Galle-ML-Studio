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
from helper.helper import _extract_feature_statistics

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
    "Logistic Regression": {
        "class": LogisticRegression,
        "params": {
            "C": {
                "type": "float",
                "default": 1.0,
                "min": 0.001,
                "max": 1000.0,
            },
            "max_iter": {
                "type": "int",
                "default": 200,
                "min": 50,
                "max": 2000,
            },
            "solver": {
                "type": "str",
                "default": "lbfgs",
                "options": ["lbfgs", "liblinear", "saga"],
            },
        },
    },
    "Random Forest Classifier": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
            },
            "max_depth": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
                "none_option": True,
            },
            "random_state": {
                "type": "int",
                "default": 42,
                "hidden": True,
            },
        },
    },
    "XGBoost Classifier": {
        "class": XGBClassifier,
        "params": {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
            },
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "min": 0.001,
                "max": 1.0,
            },
            "max_depth": {
                "type": "int",
                "default": 6,
                "min": 1,
                "max": 20,
            },
            "random_state": {
                "type": "int",
                "default": 42,
                "hidden": True,
            },
        },
    },
    "Linear Regression": {
        "class": LinearRegression,
        "params": {},
    },
    "Random Forest Regressor": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
            },
            "max_depth": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
                "none_option": True,
            },
            "random_state": {
                "type": "int",
                "default": 42,
                "hidden": True,
            },
        },
    },
    "XGBoost Regressor": {
        "class": XGBRegressor,
        "params": {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
            },
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "min": 0.001,
                "max": 1.0,
            },
            "max_depth": {
                "type": "int",
                "default": 6,
                "min": 1,
                "max": 20,
            },
            "random_state": {
                "type": "int",
                "default": 42,
                "hidden": True,
            },
        },
    },
}


# =========================================================
# FEATURE VALIDATION
# =========================================================


def validate_features_for_task(df: pd.DataFrame, features: list, task_type: str):
    """
    Validasi apakah feature yang dipilih sesuai dengan task type.

    Returns:
        (is_valid: bool, errors: list[str], warnings: list[str])
    """
    errors = []
    warnings = []

    if not features:
        errors.append("Tidak ada feature yang dipilih.")
        return False, errors, warnings

    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        errors.append(f"Kolom tidak ditemukan di dataset: {', '.join(missing_cols)}")
        return False, errors, warnings

    for col in features:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        n_unique = series.nunique()
        null_pct = series.isna().mean() * 100

        # ── NULL check ──────────────────────────────────────────────────────
        if null_pct > 80:
            warnings.append(
                f"Kolom '{col}' memiliki {null_pct:.0f}% nilai kosong — "
                f"mungkin tidak informatif."
            )

        # ── Constant column check ────────────────────────────────────────────
        if n_unique <= 1:
            warnings.append(
                f"Kolom '{col}' hanya memiliki {n_unique} nilai unik — "
                f"tidak akan membantu model."
            )

        # ── Regression: butuh setidaknya beberapa fitur numerik ─────────────
        if task_type == "regression":
            if not is_numeric:
                warnings.append(
                    f"Kolom '{col}' bukan numerik. "
                    f"Untuk Regresi, disarankan menggunakan fitur numerik. "
                    f"Kolom kategorik akan di-encode otomatis, tapi perhatikan kardinalitasnya."
                )

        # ── High cardinality categorical warning ─────────────────────────────
        if not is_numeric and n_unique > 50:
            warnings.append(
                f"Kolom '{col}' memiliki {n_unique} nilai unik (kategorik tinggi). "
                f"One-hot encoding akan menghasilkan banyak kolom — "
                f"pertimbangkan untuk drop atau encode manual."
            )

    # ── Regression: pastikan ada minimal 1 fitur numerik ────────────────────
    if task_type == "regression":
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        if len(numeric_features) == 0:
            errors.append(
                "Regresi membutuhkan minimal 1 fitur numerik. "
                "Semua fitur yang dipilih bukan numerik."
            )

    # ── Anomaly: semua fitur harus numerik ──────────────────────────────────
    # (model anomaly detection seperti IsolationForest, LOF, dll tidak bisa
    #  terima raw kategorik — preprocessor akan handle, tapi berikan warning)
    if task_type == "anomaly":
        non_numeric = [f for f in features if not pd.api.types.is_numeric_dtype(df[f])]
        if non_numeric:
            warnings.append(
                f"Fitur berikut bukan numerik: {', '.join(non_numeric)}. "
                f"Untuk Anomaly Detection, disarankan hanya pakai fitur numerik. "
                f"Kolom kategorik akan di-encode otomatis via OneHotEncoder."
            )
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        if len(numeric_features) == 0:
            errors.append(
                "Anomaly Detection membutuhkan minimal 1 fitur numerik. "
                "Semua fitur yang dipilih bukan numerik."
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_target_for_task(df: pd.DataFrame, target_col: str, task_type: str):
    """
    Validasi apakah target column sesuai dengan task type.

    Returns:
        (is_valid: bool, errors: list[str], warnings: list[str])
    """
    errors = []
    warnings = []

    if task_type == "anomaly":
        # Anomaly tidak butuh target
        return True, errors, warnings

    if not target_col or target_col not in df.columns:
        errors.append(f"Kolom target '{target_col}' tidak ditemukan di dataset.")
        return False, errors, warnings

    series = df[target_col]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    n_unique = series.nunique()
    null_pct = series.isna().mean() * 100

    if null_pct > 20:
        warnings.append(
            f"Target '{target_col}' memiliki {null_pct:.0f}% nilai kosong — "
            f"pertimbangkan untuk cleaning data terlebih dahulu."
        )

    if task_type == "regression":
        if not is_numeric:
            errors.append(
                f"Target '{target_col}' bukan numerik, tapi task type adalah Regresi. "
                f"Regresi hanya bisa memprediksi nilai angka (contoh: harga, suhu, pendapatan). "
                f"Ganti task ke Klasifikasi, atau pilih kolom target yang berisi angka."
            )
        elif n_unique < 5:
            warnings.append(
                f"Target '{target_col}' hanya memiliki {n_unique} nilai unik. "
                f"Mungkin lebih cocok untuk Klasifikasi daripada Regresi."
            )

    if task_type == "classification":
        if is_numeric and n_unique > 50:
            warnings.append(
                f"Target '{target_col}' memiliki {n_unique} nilai unik numerik. "
                f"Mungkin lebih cocok untuk Regresi daripada Klasifikasi."
            )
        if n_unique < 2:
            errors.append(
                f"Target '{target_col}' hanya memiliki {n_unique} kelas unik. "
                f"Klasifikasi membutuhkan minimal 2 kelas."
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


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

        # ── FIX: selalu inisialisasi score_stats & feature_stats ──────────────
        # Supaya tidak error "has no attribute score_stats" saat task bukan anomaly
        self.score_stats = {}
        self.feature_stats = {}

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
    # TRAIN — ENTRY POINT
    # =====================================================

    def train(self, selected_models):
        """
        Entry point training. Dispatch ke method yang sesuai task_type.
        Validasi feature & target dilakukan di sini sebelum training dimulai.
        """
        # ── Validasi target ──────────────────────────────────────────────────
        is_valid_target, target_errors, target_warnings = validate_target_for_task(
            self.df, self.target_column, self.task_type
        )
        if not is_valid_target:
            raise ValueError(
                "❌ Target column tidak valid:\n\n" + "\n".join(target_errors)
            )

        # ── Validasi features ────────────────────────────────────────────────
        is_valid_feat, feat_errors, feat_warnings = validate_features_for_task(
            self.df, self.selected_features, self.task_type
        )
        if not is_valid_feat:
            raise ValueError(
                "❌ Feature selection tidak valid:\n\n" + "\n".join(feat_errors)
            )

        # Log warnings (tidak stop training)
        for w in target_warnings + feat_warnings:
            print(f"[WARNING] {w}")

        X = self.df[self.selected_features]

        self._build_preprocessor(X)

        if self.task_type == "anomaly":
            self._train_anomaly(X, selected_models)
        elif self.task_type == "classification":
            self._train_classification(X, selected_models)
        elif self.task_type == "regression":
            self._train_regression(X, selected_models)
        else:
            raise ValueError(f"Task type tidak dikenal: '{self.task_type}'")

        return (self.results, self.all_trained_models)

    # =====================================================
    # CLASSIFICATION TRAINING
    # =====================================================

    def _train_classification(self, X, selected_models):
        print("========== CLASSIFICATION TRAINING ==========")

        y = self.df[self.target_column]

        # Label encode target jika kategorik
        if not pd.api.types.is_numeric_dtype(y):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        classification_models = get_available_models("classification")

        for model_name in selected_models:
            if model_name not in classification_models:
                print(f"SKIP (bukan classification model): {model_name}")
                continue

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
                        ("preprocessor", self.preprocessor),
                        ("classifier", model),
                    ]
                )

                start_time = time.time()
                pipeline.fit(X_train, y_train)
                duration = round(time.time() - start_time, 2)

                y_pred = pipeline.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                summary = {
                    "model_name": model_name,
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "training_duration": duration,
                    "model_configuration": actual_params,
                    "total_rows": len(X),
                    "test_rows": len(X_test),
                }

                self.results[model_name] = summary
                self.all_trained_models[model_name] = pipeline
                self.training_durations[model_name] = duration
                self.model_evaluation_details[model_name] = {"summary": summary}

                print(f"SUCCESS: {model_name} | Accuracy: {acc:.4f}")

            except Exception as e:
                print(f"ERROR TRAINING {model_name}: {e}")
                self.results[model_name] = {"error": str(e)}

        self._select_best_model_supervised("f1_score")

    # =====================================================
    # REGRESSION TRAINING
    # =====================================================

    def _train_regression(self, X, selected_models):
        print("========== REGRESSION TRAINING ==========")

        y = self.df[self.target_column]

        # ── Validasi eksplisit: target harus numerik untuk regression ─────────
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                f"❌ Target '{self.target_column}' bukan numerik.\n"
                f"Regresi hanya bisa memprediksi nilai angka.\n"
                f"Ganti task ke Klasifikasi, atau pilih target yang berisi angka."
            )

        y = y.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        regression_models = get_available_models("regression")

        for model_name in selected_models:
            if model_name not in regression_models:
                print(f"SKIP (bukan regression model): {model_name}")
                continue

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
                        ("preprocessor", self.preprocessor),
                        ("classifier", model),
                    ]
                )

                start_time = time.time()
                pipeline.fit(X_train, y_train)
                duration = round(time.time() - start_time, 2)

                y_pred = pipeline.predict(X_test)

                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = float(r2_score(y_test, y_pred))

                summary = {
                    "model_name": model_name,
                    "rmse": round(rmse, 4),
                    "r2_score": round(r2, 4),
                    "training_duration": duration,
                    "model_configuration": actual_params,
                    "total_rows": len(X),
                    "test_rows": len(X_test),
                }

                self.results[model_name] = summary
                self.all_trained_models[model_name] = pipeline
                self.training_durations[model_name] = duration
                self.model_evaluation_details[model_name] = {"summary": summary}

                print(f"SUCCESS: {model_name} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

            except Exception as e:
                print(f"ERROR TRAINING {model_name}: {e}")
                self.results[model_name] = {"error": str(e)}

        self._select_best_model_supervised("r2_score", higher_is_better=True)

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

                if scores is not None and len(scores) > 0:
                    self.score_stats = {
                        "p0_1": float(np.percentile(scores, 0.1)),
                        "p1": float(np.percentile(scores, 1)),
                        "p3": float(np.percentile(scores, 3)),
                        "mean": float(np.mean(scores)),
                    }
                else:
                    self.score_stats = {}

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
    # BEST MODEL — ANOMALY
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
    # BEST MODEL — SUPERVISED (classification & regression)
    # =====================================================

    def _select_best_model_supervised(
        self, metric_key: str, higher_is_better: bool = True
    ):
        best_model = None
        best_score = -np.inf if higher_is_better else np.inf

        for model_name, metrics in self.results.items():
            if "error" in metrics:
                continue
            score = metrics.get(metric_key, -np.inf if higher_is_better else np.inf)
            if higher_is_better and score > best_score:
                best_score = score
                best_model = model_name
            elif not higher_is_better and score < best_score:
                best_score = score
                best_model = model_name

        self.best_model_name = best_model
        print(f"BEST MODEL: {self.best_model_name} ({metric_key}: {best_score})")

    def save_best_model(self, project_name):

        if (
            not self.best_model_name
            or self.best_model_name not in self.all_trained_models
        ):
            return None

        artifacts_dir = os.path.join("projects", project_name, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(
            artifacts_dir,
            f"{self.best_model_name}_selected_model.joblib",
        )

        pipeline = _sanitize_pipeline(self.all_trained_models[self.best_model_name])

        safe_label_encoder = None
        if self.task_type != "anomaly" and self.label_encoder is not None:
            safe_label_encoder = {
                "classes": [str(c) for c in self.label_encoder.classes_]
            }

        # ── feature_statistics: ekstrak dari StandardScaler ───────────────────
        feature_statistics = _extract_feature_statistics(pipeline)

        export_payload = {
            "pipeline": pipeline,
            "label_encoder": safe_label_encoder,
            "datetime_cols": list(self.datetime_cols),
            "task_type": self.task_type,
            "feature_statistics": feature_statistics,  # ← BARU
        }

        joblib.dump(export_payload, model_path)

        print("\n======================================")
        print("MODEL EXPORT SUCCESS")
        print("======================================")
        print(f"Model Path : {model_path}")
        print(f"Task Type  : {self.task_type}")
        if feature_statistics:
            print(f"Feature Stats : {len(feature_statistics)} features tersimpan")
        print("======================================")

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

    if task_type == "classification":
        return [
            "Logistic Regression",
            "Random Forest Classifier",
            "XGBoost Classifier",
        ]

    if task_type == "regression":
        return [
            "Linear Regression",
            "Random Forest Regressor",
            "XGBoost Regressor",
        ]

    return []


def _sanitize_pipeline(pipeline):
    import numpy as np

    for _, step in pipeline.steps:

        # ── IsolationForest: hapus training cache via __dict__ langsung ──
        if hasattr(step, "estimators_"):
            for estimator in step.estimators_:
                # Hapus via __dict__ karena estimators_samples_ adalah property read-only
                estimator.__dict__.pop("estimators_samples_", None)
            step.__dict__.pop("estimators_samples_", None)

        # ── ColumnTransformer ────────────────────────────────────────────
        if not hasattr(step, "transformers_"):
            continue

        for _, transformer, _ in step.transformers_:
            if not hasattr(transformer, "steps"):
                continue
            for _, t in transformer.steps:
                if hasattr(t, "statistics_"):
                    t.statistics_ = np.array(t.statistics_)
                if hasattr(t, "categories_"):
                    t.categories_ = [np.array(c, dtype=object) for c in t.categories_]
                if hasattr(t, "mean_"):
                    t.mean_ = np.array(t.mean_)
                if hasattr(t, "scale_"):
                    t.scale_ = np.array(t.scale_)
                if hasattr(t, "var_"):
                    t.var_ = np.array(t.var_)

    return pipeline
