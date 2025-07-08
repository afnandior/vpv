
import pandas as pd
from modules.ml_pipeline import MLModelPipeline

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

features = ["tenure", "MonthlyCharges"]
target = "TotalCharges"

pipeline = MLModelPipeline(df, features, target)

pipeline.run_linear_regression()
pipeline.run_ridge_regression()
pipeline.run_lasso_regression()
pipeline.run_elasticnet_regression()
pipeline.run_decision_tree_regression()
pipeline.run_random_forest_regression()
pipeline.run_gradient_boosting_regression()
