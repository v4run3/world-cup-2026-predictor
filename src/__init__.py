"""Core package for the World Cup 2026 predictor.

Modules:
    data_collection — download/load historical results & rankings (Step 2)
    features        — Elo, form, goal stats feature engineering (Step 3)
    model           — XGBoost outcome + Poisson scoreline models (Step 4)
    predict         — high-level prediction API used by the UI (Step 5)
"""
