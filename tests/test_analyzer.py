import pandas as pd

from core.analyzer import DatasetAnalyzer

df = pd.read_csv("datasets/sample.csv")

analyzer = DatasetAnalyzer(df)

result = analyzer.analyze()

print(result)
