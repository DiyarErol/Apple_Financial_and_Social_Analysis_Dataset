"""
Feature Importance Analysis Script
Apple Financial and Social Analysis Dataset

Steps:
1) Load processed dataset with events
2) Build features (numeric only, excluding targets/leaks) and target next_day_close
3) Train/validation split: last 6 months hold-out; train uses TimeSeriesSplit CV
4) Fit LightGBM regressor and compute gain-based importance
5) Export feature_importance.csv sorted desc
6) Plot top-30 importance bar chart
7) Optionally compute SHAP summary if shap is available
8) Print top-10 features to console
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# Optional LightGBM import with graceful error
try:
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("LightGBM is required. Please install lightgbm.") from exc


# ============================================================================
# Helpers
# ============================================================================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("=" * 80)
    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def prepare_features(df: pd.DataFrame):
    print("\n[2/7] Preparing features and target...")

    target_col = 'next_day_close'
    # If target is missing, derive from close with a 1-day forward shift
    if target_col not in df.columns:
        if 'close' not in df.columns:
            raise ValueError("Neither 'next_day_close' nor 'close' found to derive target.")
        df[target_col] = df['close'].shift(-1)
        print("  Target not found; derived next_day_close via close.shift(-1)")

    # Drop obvious leak targets and non-features
    leak_cols = {
        'date', 'direction', 'next_day_close', 'next_day_return',
        'next_week_close', 'next_week_return',
    }
    leak_cols.update({col for col in df.columns if col.lower().startswith('next_')})

    feature_df = df.drop(columns=[col for col in leak_cols if col in df.columns], errors='ignore')

    # Select numeric columns only
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    X = feature_df[numeric_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is NaN (trailing row after shift)
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    df_clean = df.loc[valid_mask]

    print(f"✓ Feature matrix: {X.shape[0]:,} rows, {X.shape[1]} numeric features")
    print(f"✓ Target: {target_col} (after dropping NaNs: {valid_mask.sum():,} rows)")
    return df_clean, X, y


def train_valid_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    print("\n[3/7] Train/validation split (last 6 months hold-out)...")

    max_date = df['date'].max()
    cutoff = max_date - pd.DateOffset(months=6)

    train_idx = df['date'] < cutoff
    valid_idx = df['date'] >= cutoff

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    if len(X_valid) == 0 or len(X_train) == 0:
        raise ValueError("Train/validation split failed: insufficient data in one of the splits.")

    print(f"✓ Train: {len(X_train):,} rows | Valid: {len(X_valid):,} rows")
    print(f"  Cutoff date: {cutoff.date()}")
    return X_train, X_valid, y_train, y_valid


def cross_validate_model(X_train, y_train, n_splits=5):
    print("\n[4/7] TimeSeriesSplit cross-validation (n_splits=5)...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, r2s = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        rmse = np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds))
        r2 = r2_score(y_train.iloc[val_idx], preds)
        rmses.append(rmse)
        r2s.append(r2)
        print(f"  Fold {fold}: RMSE={rmse:.3f}, R2={r2:.3f}")

    print(f"✓ CV RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    print(f"✓ CV R2  : {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")


def train_final_model(X_train, y_train):
    print("\n[5/7] Training final LightGBM model on training data...")
    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_feature_importance(model, feature_names):
    print("\n[6/7] Computing feature importance (gain)...")
    booster = model.booster_
    importances = booster.feature_importance(importance_type='gain')
    fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    return fi


def save_importance_csv(fi: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fi.to_csv(output_path, index=False)
    print(f"✓ Saved feature importance CSV: {output_path} ({len(fi)} features)")


def plot_top_importance(fi: pd.DataFrame, output_path: Path, top_n=30):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_fi = fi.head(top_n)

    plt.figure(figsize=(10, 12))
    plt.barh(top_fi['feature'][::-1], top_fi['importance'][::-1], color='#1f77b4')
    plt.xlabel('Gain Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importance (Gain)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved top-{top_n} bar chart: {output_path}")


def try_shap_summary(model, X_valid: pd.DataFrame, output_path: Path, max_samples=500):
    print("\n[7/7] SHAP summary (optional)...")
    try:
        import shap  # type: ignore
    except ImportError:
        print("  shap not installed; skipping SHAP summary.")
        return

    # Sample validation data to keep runtime manageable
    if len(X_valid) > max_samples:
        X_sample = X_valid.tail(max_samples)
    else:
        X_sample = X_valid.copy()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=(10, 8))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ SHAP summary saved: {output_path}")
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"  ⚠ SHAP summary failed: {exc}")


def print_top_features(fi: pd.DataFrame, top_n=10):
    print("\nTop features (gain):")
    for i, row in fi.head(top_n).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.2f}")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / 'processed' / 'apple_with_events.csv'
    reports_dir = script_dir / 'reports' / 'final'
    fi_csv = reports_dir / 'feature_importance.csv'
    fi_fig = reports_dir / 'feature_importance_top30.png'
    shap_fig = reports_dir / 'shap_summary.png'

    df = load_dataset(data_path)
    df_clean, X, y = prepare_features(df)
    X_train, X_valid, y_train, y_valid = train_valid_split(df_clean, X, y)

    cross_validate_model(X_train, y_train, n_splits=5)
    model = train_final_model(X_train, y_train)

    fi = compute_feature_importance(model, X.columns.tolist())
    save_importance_csv(fi, fi_csv)
    plot_top_importance(fi, fi_fig, top_n=30)

    try_shap_summary(model, X_valid, shap_fig, max_samples=500)
    print_top_features(fi, top_n=10)

    # Final validation score on hold-out
    preds_valid = model.predict(X_valid)
    rmse_valid = np.sqrt(mean_squared_error(y_valid, preds_valid))
    r2_valid = r2_score(y_valid, preds_valid)
    print(f"\nHold-out (last 6 months) RMSE: {rmse_valid:.3f}")
    print(f"Hold-out (last 6 months) R2  : {r2_valid:.3f}")

    print("\nPipeline complete. Outputs saved to reports/final.")


if __name__ == '__main__':
    main()
