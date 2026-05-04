"""
AFMIP — Walk-Forward Backtest (Startup Grade)
==============================================
Tests all horizons with proper financial metrics.
Generates a full PDF-ready report.

Usage:
    python -m src.ml.evaluate
    python -m src.ml.evaluate --horizon 1d --windows 12
"""

import argparse
import warnings
warnings.filterwarnings("ignore")
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.ml.gold_adapter import (
    ALL_FEATURE_COLS as FEATURE_COLS,
    ALL_TARGET_COLS as TARGET_COLS,
    HORIZON_LABELS,
    TARGET_HORIZON_DAYS,
    ensure_prepared_features,
)
from src.ml.train import load_model
from src.ml.train import _build_lgbm, _build_xgb, _build_rf, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE

logger = logging.getLogger(__name__)
REPORTS_DIR = Path("data/reports")


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / 252
    return float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() > 0 else 0.0

def max_drawdown(cum_returns: pd.Series) -> float:
    roll_max  = cum_returns.cummax()
    drawdown  = (cum_returns - roll_max) / roll_max
    return float(drawdown.min())

def calmar_ratio(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    dd  = max_drawdown(cum)
    ann_return = (cum.iloc[-1] ** (252 / len(returns))) - 1
    return float(ann_return / abs(dd)) if dd != 0 else 0.0

def compute_metrics(returns: pd.Series) -> dict:
    cum = (1 + returns).cumprod()
    return {
        "total_return":    round(float(cum.iloc[-1] - 1), 4),
        "sharpe_ratio":    round(sharpe_ratio(returns), 3),
        "max_drawdown":    round(max_drawdown(cum), 4),
        "calmar_ratio":    round(calmar_ratio(returns), 3),
        "win_rate":        round(float((returns > 0).mean()), 4),
        "n_days":          int(len(returns)),
    }


# ---------------------------------------------------------------------------
# Walk-forward backtest for one horizon
# ---------------------------------------------------------------------------

def backtest_horizon(
    df: pd.DataFrame,
    target_col: str,
    n_windows: int = 10,
    confidence_threshold: float = 0.52,
) -> pd.DataFrame:

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = np.sort(df["date"].unique())

    total_months = int((dates[-1] - dates[0]) / np.timedelta64(30, "D"))
    if total_months < 24:
        raise ValueError(f"Not enough data for backtest ({total_months} months)")

    window_size   = len(dates) // (n_windows + 1)
    train_end_idx = len(dates) - (n_windows * window_size)

    daily_rows = []
    builder = _build_lgbm if LIGHTGBM_AVAILABLE else (_build_xgb if XGBOOST_AVAILABLE else _build_rf)

    for w in range(n_windows):
        ts = train_end_idx + w * window_size
        te = ts + window_size

        train_df = df[df["date"].isin(dates[:ts])].dropna(
            subset=FEATURE_COLS + [target_col]
        )
        test_df = df[df["date"].isin(dates[ts:te])].dropna(
            subset=FEATURE_COLS + [target_col]
        ).copy()

        if len(train_df) < 1000:
            continue

        pipeline = builder()
        pipeline.fit(train_df[FEATURE_COLS], train_df[target_col])

        horizon_days = TARGET_HORIZON_DAYS[target_col]
        test_df["confidence"] = pipeline.predict_proba(test_df[FEATURE_COLS])[:, 1]
        test_df["signal"] = (test_df["confidence"] >= confidence_threshold).astype(int)
        test_df["future_return"] = (
            test_df.groupby("ticker")["close"].shift(-horizon_days) / test_df["close"] - 1
        )
        test_df = test_df.dropna(subset=["future_return"])

        window_acc = (test_df["signal"] == test_df[target_col]).mean()
        t0 = pd.Timestamp(dates[ts]).strftime("%Y-%m")
        t1 = pd.Timestamp(dates[min(te, len(dates)-1)]).strftime("%Y-%m")
        avg_sigs = test_df.groupby("date")["signal"].sum().mean()

        logger.info(
            f"  [{HORIZON_LABELS[target_col]}] Window {w+1}/{n_windows}: "
            f"{t0} → {t1} | acc={window_acc:.3f} | signals/day={avg_sigs:.0f}"
        )

        for date, day in test_df.groupby("date"):
            long_pos     = day[day["signal"] == 1]
            strategy_ret = long_pos["future_return"].mean() if len(long_pos) > 0 else 0.0
            bh_ret       = day["future_return"].mean()
            daily_rows.append({
                "date":            date,
                "strategy_return": strategy_ret,
                "actual_return":   bh_ret,
                "n_signals":       len(long_pos),
                "accuracy":        (day["signal"] == day[target_col]).mean(),
                "window":          w + 1,
            })

    return pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-horizon backtest report
# ---------------------------------------------------------------------------

def run_full_backtest(
    df: pd.DataFrame,
    horizons: list | None = None,
    n_windows: int = 10,
) -> dict:

    targets = horizons or list(TARGET_COLS.keys())
    all_results = {}

    for target_col in targets:
        label = HORIZON_LABELS[target_col]
        logger.info(f"\nBacktesting {label} ...")
        try:
            daily = backtest_horizon(df, target_col, n_windows=n_windows)
            strat = compute_metrics(daily["strategy_return"])
            bh    = compute_metrics(daily["actual_return"])
            all_results[target_col] = {
                "label":    label,
                "daily":    daily,
                "strategy": strat,
                "buyhold":  bh,
            }
        except Exception as e:
            logger.warning(f"  {label} backtest failed: {e}")

    return all_results


def plot_full_report(all_results: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(all_results)
    if n == 0:
        return None

    fig = plt.figure(figsize=(16, 4 * n))
    fig.suptitle("AFMIP — Walk-Forward Backtest Report", fontsize=16, fontweight="bold", y=1.01)

    for i, (target_col, res) in enumerate(all_results.items()):
        daily  = res["daily"]
        label  = res["label"]
        strat  = res["strategy"]
        bh     = res["buyhold"]

        strat_cum = (1 + daily["strategy_return"]).cumprod()
        bh_cum    = (1 + daily["actual_return"]).cumprod()

        ax = fig.add_subplot(n, 1, i + 1)
        ax.plot(daily["date"], strat_cum.values, label="Strategy", color="#2563eb", linewidth=1.5)
        ax.plot(daily["date"], bh_cum.values,    label="Buy & Hold", color="#9ca3af", linewidth=1, linestyle="--")
        ax.axhline(1, color="black", linewidth=0.4, linestyle=":")
        ax.set_title(
            f"{label}  |  "
            f"Strategy: ret={strat['total_return']:.1%} sharpe={strat['sharpe_ratio']:.2f} dd={strat['max_drawdown']:.1%}  |  "
            f"B&H: ret={bh['total_return']:.1%} sharpe={bh['sharpe_ratio']:.2f}",
            fontsize=10,
        )
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = REPORTS_DIR / "backtest_full_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Report saved → {out}")
    return out


def evaluate_saved_models(
    df: pd.DataFrame,
    horizons: list | None = None,
) -> dict:
    """
    Evaluate saved production models on a prepared feature dataset.
    This is the path to use when Person 2 sends a newer gold dataset for testing.
    """
    results = {}
    targets = horizons or list(TARGET_COLS.keys())

    for target_col in targets:
        try:
            pipeline, metrics = load_model(target_col)
        except FileNotFoundError:
            logger.warning(f"No saved model found for {target_col}; skipping.")
            continue

        eval_df = df.dropna(subset=FEATURE_COLS + [TARGET_COLS[target_col]]).copy()
        if eval_df.empty:
            logger.warning(f"No valid rows available for {target_col}; skipping.")
            continue

        X = eval_df[FEATURE_COLS]
        y = eval_df[TARGET_COLS[target_col]]
        y_pred = pipeline.predict(X)
        y_prob = pipeline.predict_proba(X)[:, 1]

        results[target_col] = {
            "label": HORIZON_LABELS[target_col],
            "rows": int(len(eval_df)),
            "model_type": metrics.get("model_type", "unknown"),
            "trained_at": metrics.get("trained_at"),
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "roc_auc": round(roc_auc_score(y, y_prob), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "confidence_threshold": metrics.get("confidence_threshold", 0.52),
        }

    return results


def print_summary(all_results: dict):
    print(f"\n{'='*78}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*78}")
    print(
        f"{'Horizon':<14} {'Return':>8} {'Sharpe':>8} {'Drawdown':>10} "
        f"{'Win Rate':>10} {'vs B&H':>8}"
    )
    print(f"{'-'*78}")
    for target_col, res in all_results.items():
        s  = res["strategy"]
        bh = res["buyhold"]
        vs = s["total_return"] - bh["total_return"]
        print(
            f"{res['label']:<14} "
            f"{s['total_return']:>8.1%} "
            f"{s['sharpe_ratio']:>8.3f} "
            f"{s['max_drawdown']:>10.1%} "
            f"{s['win_rate']:>10.1%} "
            f"{vs:>+8.1%}"
        )
    print(f"{'='*78}\n")


def print_saved_model_summary(results: dict):
    print(f"\n{'='*86}")
    print("  SAVED MODEL TEST ON PROVIDED DATASET")
    print(f"{'='*86}")
    print(
        f"{'Horizon':<14} {'Rows':>7} {'Model':<10} {'Accuracy':>10} "
        f"{'ROC-AUC':>10} {'Precision':>10} {'Recall':>9} {'F1':>8}"
    )
    print(f"{'-'*86}")
    for target_col, row in results.items():
        print(
            f"{row['label']:<14} {row['rows']:>7} {row['model_type'].upper():<10} "
            f"{row['accuracy']:>10.4f} {row['roc_auc']:>10.4f} "
            f"{row['precision']:>10.4f} {row['recall']:>9.4f} {row['f1']:>8.4f}"
        )
    print(f"{'='*86}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        choices=["1d", "5d", "21d", "63d", "252d", "all"],
        default="all",
    )
    parser.add_argument("--windows",       type=int, default=10)
    parser.add_argument("--features-path", default="data/datasets/features.parquet")
    parser.add_argument("--gold-path",     default="data/datasets/gold_dataset.csv")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--saved-models",
        action="store_true",
        help="Evaluate saved production models on the prepared dataset instead of walk-forward retraining.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    features_path = ensure_prepared_features(
        features_path=args.features_path,
        gold_path=args.gold_path,
        rebuild=args.rebuild_features,
    )
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded: {df.shape}")

    horizons = (
        [f"target_{args.horizon}"] if args.horizon != "all"
        else list(TARGET_COLS.keys())
    )

    if args.saved_models:
        results = evaluate_saved_models(df, horizons=horizons)
        print_saved_model_summary(results)
    else:
        all_results = run_full_backtest(df, horizons=horizons, n_windows=args.windows)
        print_summary(all_results)
        plot_full_report(all_results)


if __name__ == "__main__":
    main()
