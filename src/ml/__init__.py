"""
AFMIP ML Decision Model
========================
Multi-horizon stock direction prediction system.

Modules:
    features  — feature engineering (24 indicators + sentiment)
    train     — multi-model training with versioning
    predict   — signals with confidence thresholds + export
    evaluate  — walk-forward backtesting with financial metrics
    monitor   — model health monitoring + drift detection
    azure_job — Azure ML cloud job submission
"""
