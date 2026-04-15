# ⚡ Autonomous AI Energy Trading Agent
### Cognizant Innovation Challenge — Utilities & Energy Domain

An autonomous AI planning agent that observes energy market conditions,
predicts future prices and demand using ML models, and recommends
optimal BUY / SELL / HOLD trading decisions with full explanations.

---

## Project Structure

```
energy_agent/
├── data/
│   └── generate_data.py      # Synthetic dataset generator (8,760 hours)
├── models/
│   └── predictor.py          # ML models: Linear Regression + Random Forest
├── agent/
│   └── planning_agent.py     # AI Planning Agent: Decision Tree + profit simulation
├── train.py                  # One-time training script
├── dashboard.py              # Streamlit dashboard (main app)
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data + train all models (run once)
```bash
python train.py
```
Expected output:
- Dataset: 8,760 hourly rows (1 year)
- Price model R²: ~0.54, MAE: ~₹5.5/unit
- Demand model R²: ~0.97, MAE: ~4.2 MW
- Agent Decision Tree CV accuracy: ~92%

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```
Opens at: http://localhost:8501

---

## How It Works

```
Weather + Demand + Price Data
        ↓
   Data Processing (Pandas/NumPy)
   Feature engineering: lags, rolling stats, cyclical time encoding
        ↓
   ML Prediction Layer (scikit-learn)
   ├── Linear Regression → predicted price t+1, t+2
   └── Random Forest     → predicted demand
        ↓
   AI Planning Agent (Decision Tree)
   ├── Simulate profit: SELL now / HOLD until t+1 / BUY for t+2
   ├── Select action with maximum expected profit
   ├── Calculate confidence score (tanh of profit margin)
   └── Generate plain-English explanation
        ↓
   Streamlit Dashboard
   Real-time decisions · History tracking · Performance metrics
```

---

## Key Technical Decisions

| Component | Technology | Why |
|---|---|---|
| Price prediction | Linear Regression | Interpretable, fast, works well for continuous target with engineered features |
| Demand prediction | Random Forest | Handles non-linear seasonal + weather patterns better than linear models |
| Agent decision | Decision Tree | Learned from profit simulation labels — interpretable, auditable |
| Dashboard | Streamlit | Production-quality UI in minimal code — ideal for MVP |
| Weather data | Open-Meteo API | Free, REST-based, no API key needed for real deployment |

---

## Future Scope: Reinforcement Learning Upgrade

The modular architecture supports upgrading the Decision Tree agent
to a Q-learning RL agent without changing the data or prediction layers:

```python
# Future: Replace Decision Tree with Q-table agent
class RLAgent:
    def update(self, state, action, reward, next_state):
        # reward = actual profit/loss from trade
        # Agent learns optimal policy over thousands of trades
```

---

## For Judges

**Business value:** 10–20% improvement in trading accuracy → ₹50–80 lakh/year savings per desk  
**Uniqueness:** Multi-step planning agent with explainable AI outputs — not just ML prediction  
**Implementability:** Full working MVP in 2–3 days, shown here  
**Scalability:** Modular, cloud-ready, supports RL upgrade path  
