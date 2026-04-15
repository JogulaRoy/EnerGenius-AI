"""
models/predictor.py
Trains and serves two ML models:
  - Linear Regression  → predicts future energy price
  - Random Forest      → predicts future demand
Both models support multi-step forecasting (t+1, t+2).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = "models"


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lag features and rolling stats so the model can
    learn temporal patterns (e.g. 'was price rising last 3 hours?').
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Lag features — what was the situation 1, 2, 3 hours ago?
    for lag in [1, 2, 3]:
        df[f"price_lag_{lag}"]       = df["price"].shift(lag)
        df[f"demand_lag_{lag}"]      = df["demand"].shift(lag)
        df[f"temperature_lag_{lag}"] = df["temperature"].shift(lag)

    # Rolling stats — short-term trend
    df["price_rolling_mean_3h"]  = df["price"].rolling(3).mean()
    df["demand_rolling_mean_3h"] = df["demand"].rolling(3).mean()
    df["price_rolling_std_3h"]   = df["price"].rolling(3).std()

    # Time encodings (cyclical so model understands 23:00 → 00:00 is small gap)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Future target: price 1 and 2 hours ahead
    df["price_t1"] = df["price"].shift(-1)
    df["price_t2"] = df["price"].shift(-2)

    return df.dropna()


PRICE_FEATURES = [
    "temperature", "sunlight", "wind_speed", "demand",
    "price_lag_1", "price_lag_2", "price_lag_3",
    "demand_lag_1", "demand_lag_2",
    "temperature_lag_1",
    "price_rolling_mean_3h", "price_rolling_std_3h",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend",
]

DEMAND_FEATURES = [
    "temperature", "sunlight", "wind_speed",
    "demand_lag_1", "demand_lag_2", "demand_lag_3",
    "demand_rolling_mean_3h",
    "temperature_lag_1",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend",
]


# ──────────────────────────────────────────────
# EnergyPredictor class
# ──────────────────────────────────────────────

class EnergyPredictor:
    """
    Trains two models:
      price_model  : Linear Regression on PRICE_FEATURES → predicts price_t1, price_t2
      demand_model : Random Forest      on DEMAND_FEATURES → predicts demand at t+1
    """

    def __init__(self):
        self.price_model_t1  = LinearRegression()
        self.price_model_t2  = LinearRegression()
        self.demand_model    = RandomForestRegressor(
            n_estimators=100, max_depth=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        self.scaler      = StandardScaler()
        self.is_trained  = False
        self.metrics     = {}

    # ── Train ──────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
        """
        Builds features, splits data, trains models.
        Returns dict of evaluation metrics.
        """
        df_feat = build_features(df)

        # ----- Price models -----
        X_price = df_feat[PRICE_FEATURES]
        y_t1    = df_feat["price_t1"]
        y_t2    = df_feat["price_t2"]

        X_price_scaled = self.scaler.fit_transform(X_price)

        Xp_tr, Xp_te, yt1_tr, yt1_te, yt2_tr, yt2_te = train_test_split(
            X_price_scaled, y_t1, y_t2, test_size=0.2, shuffle=False
        )

        self.price_model_t1.fit(Xp_tr, yt1_tr)
        self.price_model_t2.fit(Xp_tr, yt2_tr)

        price_metrics = {
            "price_t1_mae": round(mean_absolute_error(yt1_te, self.price_model_t1.predict(Xp_te)), 2),
            "price_t1_r2":  round(r2_score(yt1_te, self.price_model_t1.predict(Xp_te)), 3),
            "price_t2_mae": round(mean_absolute_error(yt2_te, self.price_model_t2.predict(Xp_te)), 2),
            "price_t2_r2":  round(r2_score(yt2_te, self.price_model_t2.predict(Xp_te)), 3),
        }

        # ----- Demand model -----
        X_dem  = df_feat[DEMAND_FEATURES]
        y_dem  = df_feat["demand"]   # predicts current demand given features

        Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
            X_dem, y_dem, test_size=0.2, shuffle=False
        )
        self.demand_model.fit(Xd_tr, yd_tr)

        demand_metrics = {
            "demand_mae": round(mean_absolute_error(yd_te, self.demand_model.predict(Xd_te)), 2),
            "demand_r2":  round(r2_score(yd_te, self.demand_model.predict(Xd_te)), 3),
        }

        self.metrics     = {**price_metrics, **demand_metrics}
        self.is_trained  = True

        print("\n=== Model Training Complete ===")
        for k, v in self.metrics.items():
            print(f"  {k}: {v}")

        return self.metrics

    # ── Predict ────────────────────────────────

    def predict(self, current_row: pd.DataFrame) -> dict:
        """
        Given a single row of current market data (already feature-engineered),
        returns predicted price_t1, price_t2, and demand.
        """
        if not self.is_trained:
            raise RuntimeError("Call train() before predict().")

        X_p = self.scaler.transform(current_row[PRICE_FEATURES])
        X_d = current_row[DEMAND_FEATURES]

        price_t1 = float(self.price_model_t1.predict(X_p)[0])
        price_t2 = float(self.price_model_t2.predict(X_p)[0])
        demand   = float(self.demand_model.predict(X_d)[0])

        return {
            "price_t1": round(price_t1, 2),
            "price_t2": round(price_t2, 2),
            "demand":   round(demand, 1),
        }

    # ── Persist ────────────────────────────────

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.price_model_t1, f"{MODEL_DIR}/price_model_t1.pkl")
        joblib.dump(self.price_model_t2, f"{MODEL_DIR}/price_model_t2.pkl")
        joblib.dump(self.demand_model,   f"{MODEL_DIR}/demand_model.pkl")
        joblib.dump(self.scaler,         f"{MODEL_DIR}/scaler.pkl")
        print(f"Models saved to {MODEL_DIR}/")

    def load(self):
        self.price_model_t1 = joblib.load(f"{MODEL_DIR}/price_model_t1.pkl")
        self.price_model_t2 = joblib.load(f"{MODEL_DIR}/price_model_t2.pkl")
        self.demand_model   = joblib.load(f"{MODEL_DIR}/demand_model.pkl")
        self.scaler         = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        self.is_trained     = True
        print("Models loaded.")
