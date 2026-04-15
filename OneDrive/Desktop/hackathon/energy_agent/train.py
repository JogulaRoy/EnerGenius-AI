"""
train.py
One-time training script. Run this before launching the dashboard.

Steps:
  1. Generate synthetic dataset (or load your own CSV)
  2. Train ML prediction models (price + demand)
  3. Train the AI planning agent (Decision Tree)
  4. Save all models to models/
  5. Print evaluation summary

Usage:
    python train.py
"""

import os
import sys
import pandas as pd

# Make sure imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

from data.generate_data import *    # runs the generator when imported
from models.predictor import EnergyPredictor, build_features
from agent.planning_agent import PlanningAgent


def main():
    print("=" * 50)
    print("  Energy Trading AI Agent — Training Pipeline")
    print("=" * 50)

    # ── Step 1: Load / generate data ──────────
    DATA_PATH = "data/energy_data.csv"
    if not os.path.exists(DATA_PATH):
        print("\n[1/4] Generating synthetic dataset...")
        os.makedirs("data", exist_ok=True)
        import subprocess
        subprocess.run(["python", "data/generate_data.py"])
    else:
        print(f"\n[1/4] Dataset found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"      Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    # ── Step 2: Train ML models ───────────────
    print("\n[2/4] Training ML prediction models...")
    predictor = EnergyPredictor()
    metrics   = predictor.train(df)
    predictor.save()

    print("\n  Price model performance:")
    print(f"    t+1 MAE: ₹{metrics['price_t1_mae']}/unit  |  R²: {metrics['price_t1_r2']}")
    print(f"    t+2 MAE: ₹{metrics['price_t2_mae']}/unit  |  R²: {metrics['price_t2_r2']}")
    print(f"\n  Demand model performance:")
    print(f"    MAE: {metrics['demand_mae']} MW  |  R²: {metrics['demand_r2']}")

    # ── Step 3: Build agent training data ─────
    print("\n[3/4] Preparing agent training data...")
    df_feat = build_features(df)
    print(f"      Feature rows (after lag dropna): {len(df_feat):,}")

    # ── Step 4: Train agent ───────────────────
    print("\n[4/4] Training AI Planning Agent (Decision Tree)...")
    agent    = PlanningAgent(use_learned_model=True)
    accuracy = agent.train_agent(df_feat)
    agent.save()

    print(f"\n  Agent CV accuracy: {accuracy * 100:.1f}%")

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE — all models saved to models/")
    print("=" * 50)
    print("\n  Run the dashboard with:")
    print("    streamlit run dashboard.py\n")


if __name__ == "__main__":
    main()
