import pandas as pd
import numpy as np


class DatasetAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def analyze(self):
        """Comprehensive dataset analysis"""

        analysis = {
            "shape": self.df.shape,
            "memory_mb": self.df.memory_usage(deep=True).sum() / (1024**2),
            "columns": {},
        }

        for col in self.df.columns:
            col_data = self.df[col]

            info = {
                "detected_type": self._detect_type(col_data),
                "null_count": col_data.isnull().sum(),
                "null_pct": col_data.isnull().sum() / len(col_data),
                "unique_count": col_data.nunique(),
                "unique_pct": col_data.nunique() / len(col_data),
                "dtype": str(col_data.dtype),
            }

            analysis["columns"][col] = info

        return analysis

    def _detect_type(self, series):

        if series.dtype in ["int64", "float64"]:
            return "numeric"

        elif series.nunique() == 2:
            return "binary"

        elif series.dtype == "bool":
            return "boolean"

        elif series.dtype == "object":

            if series.dropna().astype(str).str.len().mean() > 50:
                return "text"
            else:
                return "categorical"

        return "unknown"
