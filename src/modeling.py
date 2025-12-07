"""
Walk-Forward Validation Modeling Script for Apple Financial and Social Analysis Dataset.
Implements time series ML pipeline with 10-fold walk-forward validation on enhanced features.

Target: next_day_return (% return prediction)
Dataset: data/processed/apple_feature_enhanced.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 300


# ============================================================================
# 1. DATA LOADING AND TARGET ENGINEERING
# ============================================================================

def load_feature_dataset(csv_path):
    """Load enhanced feature dataset."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("=" * 80)
    print("\n[1/7] Loading enhanced dataset...")

    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        print(f"✓ Dataset loaded: {len(df):,} records, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        return df

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def engineer_target(df):
    """Create next_day_return target variable."""
    print("\n[2/7] Engineering target variable...")

    try:
        # Target: next day return
        df['next_day_close'] = df['close'].shift(-1)
        df['next_day_return'] = (df['next_day_close'] - df['close']) / df['close'] * 100

        # Drop rows with NaN targets
        initial_len = len(df)
        df = df.dropna(subset=['next_day_return']).reset_index(drop=True)
        dropped = initial_len - len(df)

        print(f"✓ Target created: next_day_return (%)")
        print(f"  Mean return: {df['next_day_return'].mean():.4f}%")
        print(f"  Std return: {df['next_day_return'].std():.4f}%")
        print(f"  Min return: {df['next_day_return'].min():.4f}%")
        print(f"  Max return: {df['next_day_return'].max():.4f}%")
        print(f"  Dropped {dropped} rows with NaN targets")
        print(f"  Final dataset: {len(df):,} samples")

        return df

    except Exception as e:
        print(f"✗ Error engineering target: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_features(df):
    """Prepare feature matrix excluding targets and metadata."""
    print("\n[3/7] Preparing features...")

    # Columns to exclude
    exclude_cols = [
        'date', 'next_day_close', 'next_day_return'
    ]

    # Select numeric features only
    feature_cols = [col for col in df.columns
                    if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]]

    X = df[feature_cols].values
    y = df['next_day_return'].values
    dates = df['date'].values

    print(f"✓ Features prepared: {len(feature_cols)} features, {len(X)} samples")
    print(f"  Feature columns: {', '.join(feature_cols[:5])}... (showing first 5)")

    return X, y, feature_cols, dates


# ============================================================================
# 2. WALK-FORWARD VALIDATION
# ============================================================================

def calculate_sharpe_ratio(returns, periods_per_year=252):
    """Calculate annualized Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return np.nan
    return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year) if np.std(returns) > 0 else np.nan


def walk_forward_validation(X, y, dates, feature_cols, n_splits=10):
    """
    Perform walk-forward (expanding window) validation.
    Each fold uses all previous data for training, next chunk for testing.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    dates : np.ndarray
        Dates for each sample
    feature_cols : list
        Feature names
    n_splits : int
        Number of folds

    Returns:
    --------
    dict
        Walk-forward results with per-fold metrics
    """
    print("\n[4/7] Performing 10-fold walk-forward validation...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    wf_results = {
        'fold': [],
        'train_start': [],
        'train_end': [],
        'test_start': [],
        'test_end': [],
        'train_size': [],
        'test_size': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'sharpe': [],
        'model': []
    }

    models_by_fold = {}

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        dates_train, dates_test = dates[train_idx], dates[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train LightGBM model
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_child_samples=20,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        sharpe = calculate_sharpe_ratio(y_test)

        # Store results
        wf_results['fold'].append(fold_idx)
        wf_results['train_start'].append(pd.Timestamp(dates_train[0]).date())
        wf_results['train_end'].append(pd.Timestamp(dates_train[-1]).date())
        wf_results['test_start'].append(pd.Timestamp(dates_test[0]).date())
        wf_results['test_end'].append(pd.Timestamp(dates_test[-1]).date())
        wf_results['train_size'].append(len(train_idx))
        wf_results['test_size'].append(len(test_idx))
        wf_results['rmse'].append(rmse)
        wf_results['mae'].append(mae)
        wf_results['r2'].append(r2)
        wf_results['sharpe'].append(sharpe)
        wf_results['model'].append(model)

        models_by_fold[fold_idx] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }

        print(f"\n  Fold {fold_idx:2d}:")
        print(f"    Train: {wf_results['train_start'][-1]} to {wf_results['train_end'][-1]} "
              f"({len(train_idx):,} samples)")
        print(f"    Test:  {wf_results['test_start'][-1]} to {wf_results['test_end'][-1]} "
              f"({len(test_idx):,} samples)")
        print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Sharpe: {sharpe:.4f}")

    return wf_results, models_by_fold


# ============================================================================
# 3. RESULTS AGGREGATION AND EXPORT
# ============================================================================

def save_walkforward_metrics(wf_results, output_path):
    """Save walk-forward metrics to CSV."""
    print("\n[5/7] Saving walk-forward metrics...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame (exclude model column)
    metrics_df = pd.DataFrame({
        'fold': wf_results['fold'],
        'train_start': wf_results['train_start'],
        'train_end': wf_results['train_end'],
        'test_start': wf_results['test_start'],
        'test_end': wf_results['test_end'],
        'train_size': wf_results['train_size'],
        'test_size': wf_results['test_size'],
        'rmse': wf_results['rmse'],
        'mae': wf_results['mae'],
        'r2': wf_results['r2'],
        'sharpe': wf_results['sharpe']
    })

    # Add summary row
    summary_row = pd.DataFrame({
        'fold': ['OVERALL'],
        'train_start': [None],
        'train_end': [None],
        'test_start': [None],
        'test_end': [None],
        'train_size': [np.mean(wf_results['train_size'])],
        'test_size': [np.mean(wf_results['test_size'])],
        'rmse': [np.mean(wf_results['rmse'])],
        'mae': [np.mean(wf_results['mae'])],
        'r2': [np.mean(wf_results['r2'])],
        'sharpe': [np.mean(wf_results['sharpe'])]
    })

    metrics_df = pd.concat([metrics_df, summary_row], ignore_index=True)

    metrics_df.to_csv(output_path, index=False)
    print(f"✓ Metrics saved: {output_path}")

    # Print summary
    print(f"\n  Summary Statistics:")
    print(f"    RMSE: {np.mean(wf_results['rmse']):.4f} ± {np.std(wf_results['rmse']):.4f}")
    print(f"    MAE:  {np.mean(wf_results['mae']):.4f} ± {np.std(wf_results['mae']):.4f}")
    print(f"    R²:   {np.mean(wf_results['r2']):.4f} ± {np.std(wf_results['r2']):.4f}")
    print(f"    Sharpe: {np.mean(wf_results['sharpe']):.4f} ± {np.std(wf_results['sharpe']):.4f}")


def plot_walkforward_rmse(wf_results, output_path):
    """Plot RMSE per fold."""
    print("\n[6/7] Plotting walk-forward RMSE...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: RMSE per fold
    ax = axes[0, 0]
    folds = wf_results['fold']
    rmse_vals = wf_results['rmse']
    ax.bar(folds, rmse_vals, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(rmse_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rmse_vals):.4f}')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('RMSE per Fold', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: MAE per fold
    ax = axes[0, 1]
    mae_vals = wf_results['mae']
    ax.bar(folds, mae_vals, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(mae_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mae_vals):.4f}')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_title('MAE per Fold', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: R² per fold
    ax = axes[1, 0]
    r2_vals = wf_results['r2']
    ax.bar(folds, r2_vals, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(r2_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_vals):.4f}')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_title('R² per Fold', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Sharpe Ratio per fold
    ax = axes[1, 1]
    sharpe_vals = wf_results['sharpe']
    ax.bar(folds, sharpe_vals, color='#d62728', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(sharpe_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sharpe_vals):.4f}')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax.set_title('Sharpe Ratio per Fold', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {output_path}")


def save_best_model(models_by_fold, output_path):
    """Save the best performing model (final fold)."""
    print("\n[7/7] Saving best walk-forward model...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use final fold model as best
    best_fold = max(models_by_fold.keys())
    best_model_dict = models_by_fold[best_fold]

    with open(output_path, 'wb') as f:
        pickle.dump(best_model_dict, f)

    print(f"✓ Model saved: {output_path}")
    print(f"  Model source: Fold {best_fold} (final fold)")
    print(f"  Contents: model, scaler, feature_cols")

# ============================================================================
# MAIN
# ============================================================================

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data' / 'processed'
    reports_dir = script_dir / 'reports'
    figures_dir = reports_dir / 'figures'
    models_dir = script_dir / 'models'

    input_csv = data_dir / 'apple_feature_enhanced.csv'
    output_metrics = reports_dir / 'final' / 'walkforward_metrics.csv'
    output_plot = figures_dir / 'walkforward_rmse.png'
    output_model = models_dir / 'best_walkforward.pkl'

    # Load data
    df = load_feature_dataset(input_csv)
    if df is None:
        return

    # Engineer target
    df = engineer_target(df)
    if df is None:
        return

    # Prepare features
    X, y, feature_cols, dates = prepare_features(df)

    # Walk-forward validation
    wf_results, models_by_fold = walk_forward_validation(X, y, dates, feature_cols, n_splits=10)

    # Save metrics
    save_walkforward_metrics(wf_results, output_metrics)

    # Plot
    plot_walkforward_rmse(wf_results, output_plot)

    # Save model
    save_best_model(models_by_fold, output_model)

    # Summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n✓ Output metrics: {output_metrics}")
    print(f"✓ Output plot: {output_plot}")
    print(f"✓ Output model: {output_model}")


if __name__ == '__main__':
    main()
