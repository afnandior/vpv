import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class MLModelPipeline:
    def __init__(self, df, features, target, test_size=0.2, random_state=42):
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        # division data
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train_model(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f" Model: {model.__class__.__name__}")
        print("MSE:", mse)
        print("R2:", r2)
        print("--------------")

        return {
            "model": model,
            "mse": mse,
            "r2": r2,
            "predictions": y_pred
        }

    def run_linear_regression(self):
        print(" Running Linear Regression...")
        model = LinearRegression()
        return self.train_model(model)

    def run_ridge_regression(self, alpha=1.0):
        print(" Running Ridge Regression...")
        model = Ridge(alpha=alpha)
        return self.train_model(model)

    def run_lasso_regression(self, alpha=0.1):
        print(" Running Lasso Regression...")
        model = Lasso(alpha=alpha)
        return self.train_model(model)

    def run_elasticnet_regression(self, alpha=0.1, l1_ratio=0.5):
        print(" Running ElasticNet Regression...")
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        return self.train_model(model)

    def run_decision_tree_regression(self):
        print(" Running Decision Tree Regression...")
        model = DecisionTreeRegressor()
        return self.train_model(model)

    def run_random_forest_regression(self, n_estimators=100):
        print(" Running Random Forest Regression...")
        model = RandomForestRegressor(n_estimators=n_estimators)
        return self.train_model(model)

    def run_gradient_boosting_regression(self, n_estimators=100):
        print(" Running Gradient Boosting Regression...")
        model = GradientBoostingRegressor(n_estimators=n_estimators)
        return self.train_model(model)
