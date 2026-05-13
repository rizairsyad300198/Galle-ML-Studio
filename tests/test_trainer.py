import pandas as pd

from core.feature_engine import FeatureEngine

from core.preprocessing_engine import PreprocessingEngine

from core.trainer import Trainer

df = pd.read_csv("datasets/sample.csv")

# -------------------------
# FEATURE ENGINEERING
# -------------------------

operations = [
    {"operation": "log", "source": "transaction_amount", "output": "log_amount"}
]

feature_engine = FeatureEngine(df)

df = feature_engine.apply_operations(operations)

# -------------------------
# PREPROCESSING
# -------------------------

prep = PreprocessingEngine(df)

prep.handle_missing_values()

prep.label_encode(["fund_code"])

df = prep.get_dataframe()

# -------------------------
# TRAINING DATA
# -------------------------

feature_columns = ["transaction_amount", "log_amount", "fund_code"]

target_column = "status_fraud"

train_df = df[feature_columns + [target_column]]

# -------------------------
# TRAIN MODELS
# -------------------------

trainer = Trainer(
    dataframe=train_df,
    target_column=target_column,
    selected_models=["random_forest", "logistic_regression"],
)

results = trainer.train()

print(results)
