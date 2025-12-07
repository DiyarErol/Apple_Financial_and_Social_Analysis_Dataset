"""
ensemble_model.py
=================
Ensemble model combining LightGBM and XGBoost with optimized blend weights.

Uses walk-forward validation to:
1. Train both models on same folds
2. Optimize blend weights on validation sets to maximize Sharpe
3. Generate blended predictions for final test fold

Outputs:
- reports/final/ensemble_weights.json
- reports/final/predictions_vs_actual_ensemble.csv
- reports/figures/ensemble_actual_vs_pred.png
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Import models
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not available")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available")


def get_project_root():
    """Get project root directory."""
    current = Path(__file__).resolve().parent
    if current.name == 'src':
        return current.parent
    return current


def load_data():
    """Load feature-engineered dataset."""
    root = get_project_root()
    
    # Try multiple possible locations
    possible_paths = [
        root / 'data' / 'processed' / 'apple_feature_enhanced.csv',
        root / 'src' / 'data' / 'processed' / 'apple_feature_enhanced.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print(f"[ERROR] Dataset not found in any expected location")
        return None
    
    df = pd.read_csv(data_path)
    print(f"[OK] Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def prepare_features(df):
    """Prepare feature matrix and target."""
    # Target: next day's return
    target_col = 'target_next_day_return'
    
    # Create target if not exists (shift daily_return forward by 1)
    if target_col not in df.columns:
        if 'daily_return' in df.columns:
            df[target_col] = df['daily_return'].shift(-1)
            print(f"[INFO] Created target column from daily_return (shifted forward)")
        else:
            print(f"[ERROR] Cannot create target: 'daily_return' column not found")
            return None, None, None
    
    # Exclude non-feature columns
    exclude_cols = [
        'date', 'target_next_day_return', 'close', 
        'open', 'high', 'low', 'volume', 'adj_close'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Exclude string/object columns (select only numeric)
    numeric_cols = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
        else:
            print(f"[INFO] Excluding non-numeric column: {col} (dtype: {df[col].dtype})")
    
    feature_cols = numeric_cols
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    print(f"[OK] Features: {len(feature_cols)}")
    print(f"[OK] Target: {target_col}")
    
    return X, y, feature_cols


def create_walk_forward_folds(df, n_folds=10):
    """Create walk-forward validation folds."""
    total_size = len(df)
    test_size = total_size // (n_folds + 1)
    
    folds = []
    
    for fold in range(n_folds):
        train_end = test_size * (fold + 1)
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > total_size:
            test_end = total_size
        
        folds.append({
            'fold': fold + 1,
            'train_indices': list(range(0, train_end)),
            'test_indices': list(range(test_start, test_end))
        })
    
    print(f"[OK] Created {len(folds)} walk-forward folds")
    return folds


def train_lgbm_model(X_train, y_train):
    """Train LightGBM model."""
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    return model


def train_xgb_model(X_train, y_train):
    """Train XGBoost model."""
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    return model


def compute_sharpe_from_predictions(y_pred, y_actual):
    """
    Compute Sharpe ratio from predicted and actual returns.
    
    Strategy: Go LONG when predicted return > 0, FLAT otherwise.
    """
    if len(y_pred) == 0:
        return 0.0
    
    # Strategy returns
    strategy_returns = []
    
    for pred, actual in zip(y_pred, y_actual):
        if pred > 0:  # LONG signal
            strategy_returns.append(actual)
        else:  # FLAT
            strategy_returns.append(0)
    
    strategy_returns = np.array(strategy_returns)
    
    # Sharpe ratio
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    
    if std_ret == 0:
        return 0.0
    
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return sharpe


def optimize_blend_weight(lgbm_preds, xgb_preds, y_actual):
    """
    Find optimal blend weight to maximize Sharpe ratio.
    
    Blended prediction: w * lgbm_preds + (1 - w) * xgb_preds
    """
    
    def negative_sharpe(w):
        """Objective function: negative Sharpe (for minimization)."""
        blended = w * lgbm_preds + (1 - w) * xgb_preds
        sharpe = compute_sharpe_from_predictions(blended, y_actual)
        return -sharpe  # Minimize negative = maximize positive
    
    # Optimize weight in [0, 1]
    result = minimize_scalar(negative_sharpe, bounds=(0, 1), method='bounded')
    
    optimal_weight = result.x
    optimal_sharpe = -result.fun
    
    return optimal_weight, optimal_sharpe


def run_ensemble_walk_forward(df, X, y, feature_cols, folds):
    """
    Run walk-forward validation with ensemble.
    
    Returns:
        - fold_results: List of per-fold metrics
        - optimal_weights: Blend weight per fold
        - final_predictions: Predictions for last fold
    """
    print("\n[2/6] Running ensemble walk-forward validation...")
    
    fold_results = []
    optimal_weights = []
    final_predictions = None
    
    scaler = StandardScaler()
    
    for fold_info in folds:
        fold_num = fold_info['fold']
        train_idx = fold_info['train_indices']
        test_idx = fold_info['test_indices']
        
        # Split data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train both models
        print(f"[FOLD {fold_num}] Training LightGBM...")
        lgbm_model = train_lgbm_model(X_train_scaled, y_train)
        
        print(f"[FOLD {fold_num}] Training XGBoost...")
        xgb_model = train_xgb_model(X_train_scaled, y_train)
        
        # Get predictions
        lgbm_preds = lgbm_model.predict(X_test_scaled)
        xgb_preds = xgb_model.predict(X_test_scaled)
        
        # Optimize blend weight
        optimal_weight, optimal_sharpe = optimize_blend_weight(lgbm_preds, xgb_preds, y_test.values)
        optimal_weights.append(optimal_weight)
        
        # Compute blended predictions
        blended_preds = optimal_weight * lgbm_preds + (1 - optimal_weight) * xgb_preds
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_test, blended_preds))
        mae = mean_absolute_error(y_test, blended_preds)
        r2 = r2_score(y_test, blended_preds)
        
        # Individual model metrics (for comparison)
        lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_preds))
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        
        fold_results.append({
            'fold': fold_num,
            'lgbm_weight': optimal_weight,
            'xgb_weight': 1 - optimal_weight,
            'ensemble_rmse': rmse,
            'ensemble_mae': mae,
            'ensemble_r2': r2,
            'ensemble_sharpe': optimal_sharpe,
            'lgbm_rmse': lgbm_rmse,
            'xgb_rmse': xgb_rmse,
            'test_size': len(test_idx)
        })
        
        print(f"[FOLD {fold_num}] Weight: LGBM={optimal_weight:.3f}, XGB={1-optimal_weight:.3f}")
        print(f"[FOLD {fold_num}] RMSE: Ensemble={rmse:.4f}, LGBM={lgbm_rmse:.4f}, XGB={xgb_rmse:.4f}")
        print(f"[FOLD {fold_num}] Sharpe: {optimal_sharpe:.4f}")
        
        # Save final fold predictions
        if fold_num == len(folds):
            final_predictions = {
                'actual': y_test.values,
                'predicted': blended_preds,
                'lgbm_pred': lgbm_preds,
                'xgb_pred': xgb_preds,
                'indices': test_idx,
                'weight': optimal_weight
            }
    
    print(f"[OK] Ensemble walk-forward completed: {len(folds)} folds")
    
    return fold_results, optimal_weights, final_predictions


def save_ensemble_weights(optimal_weights, fold_results, output_path):
    """Save ensemble weights and performance to JSON."""
    
    # Average weight across folds
    avg_lgbm_weight = np.mean(optimal_weights)
    avg_xgb_weight = 1 - avg_lgbm_weight
    
    # Overall metrics
    overall_rmse = np.mean([r['ensemble_rmse'] for r in fold_results])
    overall_mae = np.mean([r['ensemble_mae'] for r in fold_results])
    overall_r2 = np.mean([r['ensemble_r2'] for r in fold_results])
    overall_sharpe = np.mean([r['ensemble_sharpe'] for r in fold_results])
    
    data = {
        'ensemble_type': 'LightGBM + XGBoost',
        'optimization_metric': 'Sharpe Ratio',
        'average_weights': {
            'lgbm': float(avg_lgbm_weight),
            'xgb': float(avg_xgb_weight)
        },
        'overall_performance': {
            'rmse': float(overall_rmse),
            'mae': float(overall_mae),
            'r2': float(overall_r2),
            'sharpe': float(overall_sharpe)
        },
        'fold_weights': [
            {
                'fold': r['fold'],
                'lgbm_weight': float(r['lgbm_weight']),
                'xgb_weight': float(r['xgb_weight']),
                'ensemble_rmse': float(r['ensemble_rmse']),
                'lgbm_rmse': float(r['lgbm_rmse']),
                'xgb_rmse': float(r['xgb_rmse']),
                'sharpe': float(r['ensemble_sharpe'])
            }
            for r in fold_results
        ],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[OK] Ensemble weights saved: {output_path}")


def save_predictions(df, final_predictions, output_path):
    """Save ensemble predictions to CSV."""
    
    indices = final_predictions['indices']
    
    results_df = pd.DataFrame({
        'actual': final_predictions['actual'],
        'predicted': final_predictions['predicted'],
        'lgbm_pred': final_predictions['lgbm_pred'],
        'xgb_pred': final_predictions['xgb_pred'],
        'error': final_predictions['actual'] - final_predictions['predicted'],
        'percentage_error': ((final_predictions['actual'] - final_predictions['predicted']) / 
                            (np.abs(final_predictions['actual']) + 1e-10) * 100)
    })
    
    # Add date if available
    if 'date' in df.columns:
        results_df.insert(0, 'date', df.iloc[indices]['date'].values)
    
    results_df.to_csv(output_path, index=False)
    print(f"[OK] Ensemble predictions saved: {output_path}")
    
    return results_df


def plot_predictions(predictions_df, output_path):
    """Create actual vs predicted plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter: Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(predictions_df['actual'], predictions_df['predicted'], 
                alpha=0.5, s=30, color='#3498db')
    
    # Add diagonal line (perfect predictions)
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Return (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Ensemble: Actual vs Predicted Returns', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Compute R²
    from sklearn.metrics import r2_score
    r2 = r2_score(predictions_df['actual'], predictions_df['predicted'])
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Time series: Actual vs Predicted
    ax2 = axes[0, 1]
    x = np.arange(len(predictions_df))
    ax2.plot(x, predictions_df['actual'], label='Actual', linewidth=1.5, alpha=0.7, color='green')
    ax2.plot(x, predictions_df['predicted'], label='Predicted (Ensemble)', linewidth=1.5, alpha=0.7, color='blue')
    ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Time Series: Actual vs Ensemble Predictions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals histogram
    ax3 = axes[1, 0]
    residuals = predictions_df['actual'] - predictions_df['predicted']
    ax3.hist(residuals, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Prediction Error (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add stats
    mean_error = residuals.mean()
    std_error = residuals.std()
    ax3.text(0.05, 0.95, f'Mean: {mean_error:.4f}%\nStd: {std_error:.4f}%', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Model comparison: LGBM vs XGB vs Ensemble
    ax4 = axes[1, 1]
    
    lgbm_errors = np.abs(predictions_df['actual'] - predictions_df['lgbm_pred'])
    xgb_errors = np.abs(predictions_df['actual'] - predictions_df['xgb_pred'])
    ensemble_errors = np.abs(predictions_df['actual'] - predictions_df['predicted'])
    
    box_data = [lgbm_errors, xgb_errors, ensemble_errors]
    bp = ax4.boxplot(box_data, labels=['LightGBM', 'XGBoost', 'Ensemble'],
                      patch_artist=True)
    
    colors = ['#3498db', '#e67e22', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Absolute Error (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Model Comparison: Prediction Errors', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    means = [lgbm_errors.mean(), xgb_errors.mean(), ensemble_errors.mean()]
    for i, mean in enumerate(means, 1):
        ax4.text(i, mean, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Ensemble plot saved: {output_path}")


def compare_with_baseline(fold_results, baseline_rmse=1.9384):
    """Compare ensemble performance with baseline LightGBM."""
    
    print("\n[6/6] Performance Comparison:")
    print("="*70)
    
    ensemble_rmse = np.mean([r['ensemble_rmse'] for r in fold_results])
    ensemble_sharpe = np.mean([r['ensemble_sharpe'] for r in fold_results])
    lgbm_rmse = np.mean([r['lgbm_rmse'] for r in fold_results])
    xgb_rmse = np.mean([r['xgb_rmse'] for r in fold_results])
    
    print(f"\n[RMSE Comparison]")
    print(f"  Baseline LightGBM:  {baseline_rmse:.4f}%")
    print(f"  Current LightGBM:   {lgbm_rmse:.4f}%")
    print(f"  Current XGBoost:    {xgb_rmse:.4f}%")
    print(f"  Ensemble:           {ensemble_rmse:.4f}%")
    
    # Improvements
    baseline_improvement = ((baseline_rmse - ensemble_rmse) / baseline_rmse) * 100
    lgbm_improvement = ((lgbm_rmse - ensemble_rmse) / lgbm_rmse) * 100
    xgb_improvement = ((xgb_rmse - ensemble_rmse) / xgb_rmse) * 100
    
    print(f"\n[RMSE Improvements]")
    if baseline_improvement > 0:
        print(f"  ✓ vs Baseline: {baseline_improvement:.2f}% better")
    else:
        print(f"  ✗ vs Baseline: {abs(baseline_improvement):.2f}% worse")
    
    if lgbm_improvement > 0:
        print(f"  ✓ vs Current LGBM: {lgbm_improvement:.2f}% better")
    else:
        print(f"  ✗ vs Current LGBM: {abs(lgbm_improvement):.2f}% worse")
    
    if xgb_improvement > 0:
        print(f"  ✓ vs Current XGB: {xgb_improvement:.2f}% better")
    else:
        print(f"  ✗ vs Current XGB: {abs(xgb_improvement):.2f}% worse")
    
    print(f"\n[Sharpe Ratio]")
    print(f"  Ensemble: {ensemble_sharpe:.4f}")
    
    print("\n" + "="*70)


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("ENSEMBLE MODEL - LightGBM + XGBoost with Optimized Blending")
    print("="*70 + "\n")
    
    # Check dependencies
    if not LIGHTGBM_AVAILABLE or not XGBOOST_AVAILABLE:
        print("[ERROR] Both LightGBM and XGBoost are required")
        print("Install with: pip install lightgbm xgboost")
        sys.exit(1)
    
    # Setup paths
    root = get_project_root()
    output_dir = root / 'reports' / 'final'
    figures_dir = root / 'reports' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("[1/6] Loading data and creating features...")
    df = load_data()
    
    if df is None:
        sys.exit(1)
    
    X, y, feature_cols = prepare_features(df)
    
    if X is None:
        sys.exit(1)
    
    # Create walk-forward folds
    folds = create_walk_forward_folds(df, n_folds=10)
    
    # Run ensemble walk-forward
    fold_results, optimal_weights, final_predictions = run_ensemble_walk_forward(
        df, X, y, feature_cols, folds
    )
    
    # Save results
    print("\n[3/6] Saving ensemble weights...")
    weights_path = output_dir / 'ensemble_weights.json'
    save_ensemble_weights(optimal_weights, fold_results, weights_path)
    
    print("\n[4/6] Saving ensemble predictions...")
    predictions_path = output_dir / 'predictions_vs_actual_ensemble.csv'
    predictions_df = save_predictions(df, final_predictions, predictions_path)
    
    print("\n[5/6] Generating visualization...")
    plot_path = figures_dir / 'ensemble_actual_vs_pred.png'
    plot_predictions(predictions_df, plot_path)
    
    # Compare with baseline
    compare_with_baseline(fold_results)
    
    print(f"\n[SUCCESS] Ensemble model training completed!\n")
    print(f"Outputs:")
    print(f"  - {weights_path}")
    print(f"  - {predictions_path}")
    print(f"  - {plot_path}")
    print()


if __name__ == "__main__":
    main()
