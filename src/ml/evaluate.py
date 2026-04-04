"""
AFMIP — Stage 4: Backtesting & Evaluation
==========================================
Walk-forward backtest: train on historical data, predict on the next
window, advance the window, repeat. Computes financial metrics.

Financial metrics:
  - Cumulative return vs Buy & Hold (equal-weight portfolio)
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Hit Rate per window

Usage:
    python -m src.ml.evaluate
    python -m src.ml.evaluate --model xgboost --windows 12
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.features import FEATURE_COLS, TARGET_COL
from src.ml.train import _build_rf_pipeline, _build_xgb_pipeline, XGBOOST_AVAILABLE

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")

# Only trade when model is this confident (filters out ~50/50 guesses)
CONFIDENCE_THRESHOLD = 0.52


# ---------------------------------------------------------------------------
# Financial performance metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = daily_returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def max_drawdown(cumulative_returns: pd.Series) -> float:
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return float(drawdown.min())


def compute_metrics(returns: pd.Series) -> dict:
    cumulative = (1 + returns).cumprod()
    return {
        "total_return": float(cumulative.iloc[-1] - 1),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(cumulative),
        "win_rate":     float((returns > 0).mean()),
        "n_trades":     int(len(returns)),
    }


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    model_type: str = "rf",
    n_windows: int = 12,
    min_train_months: int = 12,
) -> pd.DataFrame:
    """
    Walk-forward backtest.

    For each window:
      - Train on all dates BEFORE the window (no lookahead)
      - Predict on the window
      - Build daily equal-weight portfolio from tickers where
        model confidence >= CONFIDENCE_THRESHOLD
      - Compare to buy-and-hold (all tickers, every day)

    Returns daily portfolio-level DataFrame (one row per date).
    """
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = np.sort(df["date"].unique())

    total_months = int((dates[-1] - dates[0]) / np.timedelta64(30, "D"))
    if total_months < min_train_months + n_windows:
        raise ValueError(
            f"Not enough data. Need at least {min_train_months + n_windows} months, "
            f"got ~{total_months}."
        )

    window_size   = len(dates) // (n_windows + 1)
    train_end_idx = len(dates) - (n_windows * window_size)

    all_daily = []

    for w in range(n_windows):
        test_start = train_end_idx + w * window_size
        test_end   = test_start + window_size

        train_dates = dates[:test_start]
        test_dates  = dates[test_start:test_end]

        train_df = df[df["date"].isin(train_dates)]
        test_df  = df[df["date"].isin(test_dates)].copy()

        if len(train_df) < 1000:
            logger.warning(f"Window {w+1}: not enough training data, skipping.")
            continue

        # Fit model
        pipeline = _build_rf_pipeline() if model_type == "rf" else _build_xgb_pipeline()
        pipeline.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

        # Predict
        test_df["confidence"] = pipeline.predict_proba(test_df[FEATURE_COLS])[:, 1]
        test_df["signal"]     = (test_df["confidence"] >= CONFIDENCE_THRESHOLD).astype(int)

        # next_return = tomorrow's return for each ticker (shift per ticker)
        test_df["next_return"] = test_df.groupby("ticker")["return_1d"].shift(-1)
        test_df = test_df.dropna(subset=["next_return"])

        # Build daily equal-weight portfolio
        for date, day in test_df.groupby("date"):
            long_pos = day[day["signal"] == 1]
            strategy_ret = long_pos["next_return"].mean() if len(long_pos) > 0 else 0.0
            bh_ret       = day["next_return"].mean()
            accuracy     = (day["signal"] == day[TARGET_COL]).mean()

            all_daily.append({
                "date":            date,
                "strategy_return": strategy_ret,
                "actual_return":   bh_ret,
                "n_signals":       len(long_pos),
                "accuracy":        accuracy,
                "window":          w + 1,
            })

        window_acc = (test_df["signal"] == test_df[TARGET_COL]).mean()
        avg_sigs   = test_df.groupby("date")["signal"].sum().mean()
        t0 = pd.Timestamp(test_dates[0]).strftime('%Y-%m')
        t1 = pd.Timestamp(test_dates[-1]).strftime('%Y-%m')
        logger.info(
            f"Window {w+1}/{n_windows}: "
            f"{t0} → {t1} | "
            f"Accuracy: {window_acc:.3f} | Avg signals/day: {avg_sigs:.0f}"
        )

    if not all_daily:
        raise RuntimeError("No valid windows produced results.")

    return pd.DataFrame(all_daily).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_cumulative_returns(daily: pd.DataFrame, model_type: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    strategy_cum = (1 + daily["strategy_return"]).cumprod()
    buyhold_cum  = (1 + daily["actual_return"]).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    axes[0].plot(daily["date"], strategy_cum.values, label="Strategy (ML signals)", color="#2563eb", linewidth=1.5)
    axes[0].plot(daily["date"], buyhold_cum.values,  label="Buy & Hold (all tickers)", color="#9ca3af", linewidth=1.5, linestyle="--")
    axes[0].axhline(1, color="black", linewidth=0.5, linestyle=":")
    axes[0].set_title(f"AFMIP — Walk-Forward Backtest ({model_type.upper()})", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Cumulative Return (×)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rolling_max = strategy_cum.cummax()
    drawdown = (strategy_cum - rolling_max) / rolling_max
    axes[1].fill_between(daily["date"], drawdown.values, 0, color="#ef4444", alpha=0.5, label="Strategy Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(daily["date"], daily["n_signals"], color="#6366f1", alpha=0.6, width=1)
    axes[2].set_ylabel("Tickers traded per day")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = REPORTS_DIR / f"backtest_{model_type}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Chart saved → {out}")
    return out


def print_summary(daily: pd.DataFrame, model_type: str):
    strat = compute_metrics(daily["strategy_return"])
    bh    = compute_metrics(daily["actual_return"])

    print(f"\n{'='*58}")
    print(f"  BACKTEST RESULTS — {model_type.upper()}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD:.0%} | Avg signals/day: {daily['n_signals'].mean():.0f}")
    print(f"{'='*58}")
    print(f"{'Metric':<24} {'Strategy':>14} {'Buy & Hold':>14}")
    print(f"{'-'*58}")
    print(f"{'Total Return':<24} {strat['total_return']:>13.2%} {bh['total_return']:>13.2%}")
    print(f"{'Sharpe Ratio':<24} {strat['sharpe_ratio']:>14.3f} {bh['sharpe_ratio']:>14.3f}")
    print(f"{'Max Drawdown':<24} {strat['max_drawdown']:>13.2%} {bh['max_drawdown']:>13.2%}")
    print(f"{'Win Rate':<24} {strat['win_rate']:>13.2%} {bh['win_rate']:>13.2%}")
    print(f"{'Avg Daily Accuracy':<24} {daily['accuracy'].mean():>13.2%} {'—':>14}")
    print(f"{'Trading Days':<24} {strat['n_trades']:>14,} {bh['n_trades']:>14,}")
    print(f"{'='*58}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run AFMIP walk-forward backtest")
    parser.add_argument("--model",         choices=["rf", "xgboost"], default="rf")
    parser.add_argument("--windows",       type=int, default=12)
    parser.add_argument("--features-path", default="data/datasets/features.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_parquet(args.features_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded features: {df.shape}")

    daily = walk_forward_backtest(df, model_type=args.model, n_windows=args.windows)

    print_summary(daily, args.model)
    plot_cumulative_returns(daily, args.model)

    out = REPORTS_DIR / f"backtest_{args.model}_results.parquet"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out, index=False)
    logger.info(f"Raw results saved → {out}")


if __name__ == "__main__":
    main()
