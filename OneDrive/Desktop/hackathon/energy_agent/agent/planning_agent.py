"""
agent/planning_agent.py

The AI Planning Agent — the core of this system.

How it works:
  1. Receives current market state + ML predictions (price_t1, price_t2, demand)
  2. Simulates profit for each possible action (BUY / SELL / HOLD)
  3. Selects the action with maximum expected profit
  4. Calculates a confidence score based on margin vs alternatives
  5. Generates a plain-English explanation

The agent is also trainable: a Decision Tree learns from historical
(current_price, predicted_price, demand) → action labels derived from
the same profit logic. This makes the agent a *learning* classifier,
not a hardcoded if-else system.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os

AGENT_MODEL_PATH = "models/agent_model.pkl"

# Thresholds — tunable
HIGH_DEMAND_THRESHOLD = 220    # MW — above this, demand is "high"
PROFIT_MARGIN_THRESHOLD = 3.0  # ₹/unit — minimum meaningful profit gap
BUY_COST_ASSUMPTION = 20       # ₹/unit — assumed base cost when buying


# ──────────────────────────────────────────────
# Core Planning Logic
# ──────────────────────────────────────────────

def simulate_profits(
    current_price: float,
    price_t1: float,
    price_t2: float,
    demand: float,
) -> dict:
    """
    Simulates the expected profit for each action.

    SELL now  : profit = current_price - BUY_COST_ASSUMPTION
                (sell immediately at current market price)

    HOLD t+1  : profit = price_t1 - BUY_COST_ASSUMPTION
                (wait one period, sell at predicted t+1 price)

    BUY now   : profit = price_t2 - current_price
                (buy now, sell later at t+2 — captures rising price)

    Demand multiplier: high demand amplifies sell/hold profit
    because grid pressure pushes prices higher.
    """
    demand_factor = 1.15 if demand > HIGH_DEMAND_THRESHOLD else 1.0

    sell_profit = (current_price - BUY_COST_ASSUMPTION) * demand_factor
    hold_profit = (price_t1 - BUY_COST_ASSUMPTION) * demand_factor
    buy_profit  = price_t2 - current_price   # directional gain

    return {
        "SELL": round(sell_profit, 2),
        "HOLD": round(hold_profit, 2),
        "BUY":  round(buy_profit, 2),
    }


def calculate_confidence(profits: dict, best_action: str) -> float:
    """
    Confidence = how much better is the best action vs the second-best?

    Formula:
        margin = best_profit - second_best_profit
        confidence = tanh(margin / PROFIT_MARGIN_THRESHOLD)
        → gives a smooth 0–1 value:
           margin = 0  → confidence = 0.0
           margin = 3  → confidence ≈ 0.76
           margin = 6  → confidence ≈ 0.96
           margin > 10 → confidence ≈ 0.99
    """
    sorted_profits = sorted(profits.values(), reverse=True)
    best   = sorted_profits[0]
    second = sorted_profits[1] if len(sorted_profits) > 1 else 0

    margin = max(best - second, 0)
    confidence = float(np.tanh(margin / PROFIT_MARGIN_THRESHOLD))
    return round(confidence, 3)


def generate_explanation(
    action: str,
    current_price: float,
    price_t1: float,
    price_t2: float,
    demand: float,
    profits: dict,
    confidence: float,
) -> str:
    """
    Produces a human-readable explanation of the agent's reasoning.
    This is the 'explainable AI' feature — every decision is transparent.
    """
    demand_str  = "high" if demand > HIGH_DEMAND_THRESHOLD else "moderate"
    trend_t1    = "rising" if price_t1 > current_price else "falling"
    trend_t2    = "rising further" if price_t2 > price_t1 else "stabilising"
    conf_pct    = f"{round(confidence * 100)}%"
    best_profit = profits[action]

    base = (
        f"Current price ₹{current_price:.1f}/unit. "
        f"Demand is {demand_str} ({demand:.0f} MW). "
        f"Price forecast: t+1 = ₹{price_t1:.1f} ({trend_t1}), "
        f"t+2 = ₹{price_t2:.1f} ({trend_t2}). "
    )

    if action == "SELL":
        reason = (
            f"Selling now yields highest expected profit of ₹{best_profit:.1f}/unit. "
            f"{'High demand amplifies current price.' if demand > HIGH_DEMAND_THRESHOLD else 'Prices expected to drop — selling now locks in value.'}"
        )
    elif action == "HOLD":
        reason = (
            f"Waiting until t+1 is expected to yield ₹{best_profit:.1f}/unit — "
            f"better than selling now (₹{profits['SELL']:.1f}) "
            f"or buying (₹{profits['BUY']:.1f}). Price is {trend_t1}."
        )
    else:  # BUY
        reason = (
            f"Buying now and selling at t+2 (₹{price_t2:.1f}) "
            f"yields estimated ₹{best_profit:.1f}/unit gain. "
            f"Market shows upward trajectory."
        )

    alert = ""
    if confidence < 0.4:
        alert = " ⚠ Low confidence — market signals are mixed. Consider manual review."
    elif confidence > 0.85:
        alert = " ✓ High confidence — strong signal alignment."

    return base + reason + f" Confidence: {conf_pct}." + alert


# ──────────────────────────────────────────────
# PlanningAgent class
# ──────────────────────────────────────────────

class PlanningAgent:
    """
    The AI Planning Agent.

    Two modes:
      - rule_mode  : uses simulate_profits() directly (always available)
      - learn_mode : uses a trained Decision Tree (requires train_agent())

    In practice both are used: the DT is trained on labels generated by
    the rule logic, so the DT learns the *same* strategy but can
    generalise and be inspected as a learned model.
    """

    def __init__(self, use_learned_model: bool = True):
        self.use_learned_model = use_learned_model
        self.dt_classifier     = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced",
        )
        self.is_trained = False
        self.cv_accuracy = None

    # ── Train the Decision Tree agent ──────────

    def train_agent(self, df_features: pd.DataFrame) -> float:
        """
        Generates BUY/SELL/HOLD labels using the profit simulation,
        then trains a Decision Tree to learn that mapping from raw features.

        This is the key step that makes the agent a *learning* system.
        """
        rows = []
        for _, row in df_features.iterrows():
            profits = simulate_profits(
                current_price=row["price"],
                price_t1=row.get("price_t1", row["price"] * 1.02),
                price_t2=row.get("price_t2", row["price"] * 1.04),
                demand=row["demand"],
            )
            best_action = max(profits, key=profits.get)
            rows.append({
                "price":           row["price"],
                "demand":          row["demand"],
                "temperature":     row["temperature"],
                "price_lag_1":     row.get("price_lag_1", row["price"]),
                "demand_lag_1":    row.get("demand_lag_1", row["demand"]),
                "price_t1_approx": row.get("price_t1", row["price"]),
                "price_t2_approx": row.get("price_t2", row["price"]),
                "action":          best_action,
            })

        df_agent = pd.DataFrame(rows).dropna()
        feature_cols = [c for c in df_agent.columns if c != "action"]

        X = df_agent[feature_cols]
        y = df_agent["action"]

        scores = cross_val_score(self.dt_classifier, X, y, cv=5, scoring="accuracy")
        self.cv_accuracy = round(float(scores.mean()), 3)

        self.dt_classifier.fit(X, y)
        self.is_trained = True

        print(f"\n=== Agent Training Complete ===")
        print(f"  Decision Tree CV accuracy: {self.cv_accuracy}")
        print(f"  Action distribution:\n{y.value_counts().to_string()}")

        return self.cv_accuracy

    # ── Make a decision ────────────────────────

    def decide(
        self,
        current_price: float,
        price_t1: float,
        price_t2: float,
        demand: float,
        temperature: float = 25.0,
        price_lag_1: float = None,
        demand_lag_1: float = None,
    ) -> dict:
        """
        Core agent decision function.
        Returns action, profits, confidence, and explanation.
        """
        if price_lag_1  is None: price_lag_1  = current_price
        if demand_lag_1 is None: demand_lag_1 = demand

        profits = simulate_profits(current_price, price_t1, price_t2, demand)

        # Decide: learned DT if available, otherwise rule-based
        if self.use_learned_model and self.is_trained:
            X = pd.DataFrame([{
                "price":           current_price,
                "demand":          demand,
                "temperature":     temperature,
                "price_lag_1":     price_lag_1,
                "demand_lag_1":    demand_lag_1,
                "price_t1_approx": price_t1,
                "price_t2_approx": price_t2,
            }])
            action = self.dt_classifier.predict(X)[0]
        else:
            action = max(profits, key=profits.get)

        confidence  = calculate_confidence(profits, action)
        explanation = generate_explanation(
            action, current_price, price_t1, price_t2,
            demand, profits, confidence
        )

        return {
            "action":      action,
            "profits":     profits,
            "confidence":  confidence,
            "confidence_pct": f"{round(confidence * 100)}%",
            "explanation": explanation,
            "price_t1":    price_t1,
            "price_t2":    price_t2,
            "demand":      demand,
        }

    # ── Persist ────────────────────────────────

    def save(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.dt_classifier, AGENT_MODEL_PATH)
        print(f"Agent model saved to {AGENT_MODEL_PATH}")

    def load(self):
        self.dt_classifier = joblib.load(AGENT_MODEL_PATH)
        self.is_trained    = True
        print("Agent model loaded.")
