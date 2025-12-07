"""
optimize_thresholds.py
======================
Grid search optimization for backtest strategy thresholds.

Searches over:
- Long threshold: [0.25%, 0.5%, 0.75%, 1.0%]
- Short threshold: [-0.25%, -0.5%, -0.75%, -1.0%]
- Transaction cost: [0.1%, 0.2%, 0.3%]

Outputs:
- reports/final/threshold_grid_results.csv (all combinations)
- reports/final/threshold_best.json (best parameters)
- reports/figures/threshold_sharpe_surface.png
- reports/figures/threshold_mdd_surface.png
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
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """Get project root directory."""
    current = Path(__file__).resolve().parent
    if current.name == 'src':
        return current.parent
    return current


def load_predictions_data():
    """
    Load predictions data. Priority:
    1) Use existing predictions_vs_actual.csv if available
    2) Generate from model if CSV doesn't exist
    
    Returns DataFrame with: date, close, actual_return, predicted_return
    """
    root = get_project_root()
    
    # Try multiple possible locations
    possible_paths = [
        root / 'predictions_vs_actual.csv',
        root / 'reports' / 'final' / 'predictions_vs_actual.csv',
        root / 'src' / 'reports' / 'final' / 'predictions_vs_actual.csv'
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break
    
    # Try loading existing CSV
    if csv_path and csv_path.exists():
        print(f"[INFO] Using existing predictions: {csv_path.name}")
        df = pd.read_csv(csv_path)
        
        # Check if we have required columns (date is optional)
        required = ['actual', 'predicted']
        if not all(col in df.columns for col in required):
            print(f"[WARNING] CSV missing required columns. Expected: {required}")
            print(f"[INFO] Available columns: {list(df.columns)}")
            return None
        
        # Add date column if missing (use index as date surrogate)
        if 'date' not in df.columns:
            df['date'] = pd.date_range(start='2015-01-01', periods=len(df), freq='D')
        
        # Convert prices to returns
        actual_prices = df['actual'].values
        predicted_prices = df['predicted'].values
        
        actual_returns = np.zeros(len(actual_prices))
        predicted_returns = np.zeros(len(predicted_prices))
        
        for i in range(len(actual_prices) - 1):
            actual_returns[i] = ((actual_prices[i+1] - actual_prices[i]) / actual_prices[i]) * 100
            predicted_returns[i] = ((predicted_prices[i+1] - predicted_prices[i]) / predicted_prices[i]) * 100
        
        # Last row has no next price, set to 0
        actual_returns[-1] = 0
        predicted_returns[-1] = 0
        
        results_df = pd.DataFrame({
            'date': df['date'],
            'close': actual_prices,
            'actual_return': actual_returns,
            'predicted_return': predicted_returns
        })
        
        # Remove last row with 0 returns
        results_df = results_df.iloc[:-1].copy()
        
        print(f"[OK] Loaded {len(results_df)} prediction samples (converted prices to returns)")
        return results_df
    
    print(f"[WARNING] predictions_vs_actual.csv not found in any expected location")
    return None


def execute_backtest(df, long_thr, short_thr, tx_cost, initial_capital=100000):
    """
    Execute backtest with given thresholds.
    
    Strategy:
    - LONG if predicted_return > long_thr
    - SHORT if predicted_return < short_thr
    - FLAT otherwise
    
    Args:
        df: DataFrame with actual_return, predicted_return
        long_thr: Long threshold (e.g., 0.005 = 0.5%)
        short_thr: Short threshold (e.g., -0.005 = -0.5%)
        tx_cost: Transaction cost per trade (e.g., 0.001 = 0.1%)
        initial_capital: Starting capital
    
    Returns:
        dict with equity curve and trade log
    """
    n = len(df)
    equity = np.zeros(n)
    equity[0] = initial_capital
    
    positions = []  # LONG=1, SHORT=-1, FLAT=0
    trades_log = []
    
    current_position = 0  # Start flat
    
    for i in range(n):
        pred = df['predicted_return'].iloc[i] / 100.0  # Convert to decimal
        actual = df['actual_return'].iloc[i] / 100.0
        
        # Determine position based on thresholds
        if pred > long_thr:
            new_position = 1  # LONG
        elif pred < short_thr:
            new_position = -1  # SHORT
        else:
            new_position = 0  # FLAT
        
        # Calculate position return
        if current_position == 1:  # Was LONG
            pos_return = actual
        elif current_position == -1:  # Was SHORT
            pos_return = -actual
        else:  # Was FLAT
            pos_return = 0.0
        
        # Apply transaction cost if position changes
        cost = 0.0
        if new_position != current_position:
            # Cost for closing old position (if not flat)
            if current_position != 0:
                cost += tx_cost
            # Cost for opening new position (if not flat)
            if new_position != 0:
                cost += tx_cost
            
            trades_log.append({
                'index': i,
                'from_pos': current_position,
                'to_pos': new_position,
                'cost': cost
            })
        
        # Update equity
        if i == 0:
            equity[i] = initial_capital
        else:
            equity[i] = equity[i-1] * (1 + pos_return - cost)
        
        positions.append(new_position)
        current_position = new_position
    
    return {
        'equity': equity,
        'positions': positions,
        'trades': trades_log
    }


def compute_metrics(equity, trades, n_days):
    """
    Compute performance metrics from equity curve.
    
    Returns dict with:
        total_return, annualized_return, volatility_annual,
        sharpe_ratio, sortino_ratio, max_drawdown,
        win_rate, trades_count, avg_holding_days
    """
    initial = equity[0]
    final = equity[-1]
    
    # Returns
    total_return = ((final / initial) - 1) * 100
    annualized_return = (((final / initial) ** (252 / n_days)) - 1) * 100
    
    # Daily returns
    daily_returns = np.diff(equity) / equity[:-1]
    
    # Volatility
    volatility_daily = np.std(daily_returns)
    volatility_annual = volatility_daily * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming 0 risk-free rate)
    mean_return = np.mean(daily_returns)
    sharpe_ratio = (mean_return / volatility_daily) * np.sqrt(252) if volatility_daily > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
    sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    max_drawdown = np.min(drawdown)
    
    # Win rate
    winning_days = np.sum(daily_returns > 0)
    total_trading_days = len(daily_returns)
    win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0
    
    # Trades count
    trades_count = len(trades)
    
    # Average holding days
    if trades_count > 1:
        holding_periods = []
        for i in range(len(trades) - 1):
            holding_periods.append(trades[i+1]['index'] - trades[i]['index'])
        avg_holding_days = np.mean(holding_periods) if holding_periods else 0
    else:
        avg_holding_days = n_days
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility_annual': volatility_annual,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades_count': trades_count,
        'avg_holding_days': avg_holding_days
    }


def grid_search(df):
    """
    Run grid search over threshold combinations.
    
    Returns DataFrame with all results.
    """
    print("\n[2/6] Running grid search...")
    
    # Grid parameters
    long_thresholds = [0.0025, 0.005, 0.0075, 0.01]  # 0.25%, 0.5%, 0.75%, 1.0%
    short_thresholds = [-0.0025, -0.005, -0.0075, -0.01]  # -0.25%, -0.5%, -0.75%, -1.0%
    tx_costs = [0.001, 0.002, 0.003]  # 0.1%, 0.2%, 0.3%
    
    total_combinations = len(long_thresholds) * len(short_thresholds) * len(tx_costs)
    print(f"[INFO] Testing {total_combinations} parameter combinations...")
    
    results = []
    
    for i, (long_thr, short_thr, tx_cost) in enumerate(product(long_thresholds, short_thresholds, tx_costs), 1):
        # Run backtest
        backtest_result = execute_backtest(df, long_thr, short_thr, tx_cost)
        
        # Compute metrics
        metrics = compute_metrics(
            backtest_result['equity'],
            backtest_result['trades'],
            len(df)
        )
        
        # Store results
        result = {
            'long_threshold': long_thr * 100,  # Convert to percentage
            'short_threshold': short_thr * 100,
            'tx_cost': tx_cost * 100,
            **metrics
        }
        results.append(result)
        
        # Progress
        if i % 12 == 0 or i == total_combinations:
            print(f"[PROGRESS] {i}/{total_combinations} combinations tested")
    
    results_df = pd.DataFrame(results)
    print(f"[OK] Grid search completed: {len(results_df)} results")
    
    return results_df


def save_results(results_df, output_dir):
    """Save grid search results to CSV and best parameters to JSON."""
    print("\n[3/6] Saving results...")
    
    # Sort by Sharpe ratio (descending)
    results_sorted = results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    
    # Save CSV
    csv_path = output_dir / 'threshold_grid_results.csv'
    results_sorted.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"[OK] Grid results saved: {csv_path}")
    
    # Save best parameters
    best = results_sorted.iloc[0].to_dict()
    json_path = output_dir / 'threshold_best.json'
    
    with open(json_path, 'w') as f:
        json.dump({
            'best_parameters': {
                'long_threshold': best['long_threshold'],
                'short_threshold': best['short_threshold'],
                'tx_cost': best['tx_cost']
            },
            'performance': {
                'total_return': best['total_return'],
                'annualized_return': best['annualized_return'],
                'sharpe_ratio': best['sharpe_ratio'],
                'sortino_ratio': best['sortino_ratio'],
                'max_drawdown': best['max_drawdown'],
                'win_rate': best['win_rate'],
                'trades_count': int(best['trades_count']),
                'avg_holding_days': best['avg_holding_days']
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print(f"[OK] Best parameters saved: {json_path}")
    
    return results_sorted


def plot_surfaces(results_df, figures_dir):
    """Create surface plots for Sharpe and Max Drawdown vs thresholds."""
    print("\n[4/6] Generating surface plots...")
    
    # For each transaction cost, create a heatmap
    tx_costs = results_df['tx_cost'].unique()
    
    # Sharpe Ratio Surface
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Sharpe Ratio vs Trading Thresholds', fontsize=16, fontweight='bold')
    
    for i, tx_cost in enumerate(sorted(tx_costs)):
        subset = results_df[results_df['tx_cost'] == tx_cost]
        
        # Pivot for heatmap
        pivot = subset.pivot_table(
            values='sharpe_ratio',
            index='short_threshold',
            columns='long_threshold'
        )
        
        # Plot
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            ax=axes[i],
            cbar_kws={'label': 'Sharpe Ratio'}
        )
        axes[i].set_title(f'Transaction Cost: {tx_cost:.2f}%')
        axes[i].set_xlabel('Long Threshold (%)')
        axes[i].set_ylabel('Short Threshold (%)')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    sharpe_path = figures_dir / 'threshold_sharpe_surface.png'
    plt.savefig(sharpe_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Sharpe surface plot saved: {sharpe_path}")
    
    # Max Drawdown Surface
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Max Drawdown vs Trading Thresholds', fontsize=16, fontweight='bold')
    
    for i, tx_cost in enumerate(sorted(tx_costs)):
        subset = results_df[results_df['tx_cost'] == tx_cost]
        
        # Pivot for heatmap
        pivot = subset.pivot_table(
            values='max_drawdown',
            index='short_threshold',
            columns='long_threshold'
        )
        
        # Plot (note: more negative = worse)
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',  # Reversed: green for less negative
            ax=axes[i],
            cbar_kws={'label': 'Max Drawdown (%)'}
        )
        axes[i].set_title(f'Transaction Cost: {tx_cost:.2f}%')
        axes[i].set_xlabel('Long Threshold (%)')
        axes[i].set_ylabel('Short Threshold (%)')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    mdd_path = figures_dir / 'threshold_mdd_surface.png'
    plt.savefig(mdd_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Max drawdown surface plot saved: {mdd_path}")


def print_top_results(results_df, n=5):
    """Print top N parameter sets to console."""
    print(f"\n[5/6] Top {n} Parameter Sets (by Sharpe Ratio):")
    print("="*100)
    
    for i, row in results_df.head(n).iterrows():
        print(f"\n[Rank {i+1}]")
        print(f"  Parameters:")
        print(f"    Long Threshold:  {row['long_threshold']:>6.2f}%")
        print(f"    Short Threshold: {row['short_threshold']:>6.2f}%")
        print(f"    TX Cost:         {row['tx_cost']:>6.2f}%")
        print(f"  Performance:")
        print(f"    Total Return:    {row['total_return']:>8.2f}%")
        print(f"    Annual Return:   {row['annualized_return']:>8.2f}%")
        print(f"    Sharpe Ratio:    {row['sharpe_ratio']:>8.4f}")
        print(f"    Sortino Ratio:   {row['sortino_ratio']:>8.4f}")
        print(f"    Max Drawdown:    {row['max_drawdown']:>8.2f}%")
        print(f"    Win Rate:        {row['win_rate']:>8.2f}%")
        print(f"    Trades:          {int(row['trades_count']):>8d}")
        print(f"    Avg Hold Days:   {row['avg_holding_days']:>8.1f}")
    
    print("\n" + "="*100)


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION - Grid Search for Backtest Strategy")
    print("="*70 + "\n")
    
    # Setup paths
    root = get_project_root()
    output_dir = root / 'reports' / 'final'
    figures_dir = root / 'reports' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("[1/6] Loading predictions and actuals...")
    df = load_predictions_data()
    
    if df is None or len(df) == 0:
        print("\n[ERROR] Failed to load predictions data")
        print("Please ensure predictions_vs_actual.csv exists or model can generate predictions")
        sys.exit(1)
    
    # Grid search
    results_df = grid_search(df)
    
    # Save results
    results_sorted = save_results(results_df, output_dir)
    
    # Plot surfaces
    plot_surfaces(results_sorted, figures_dir)
    
    # Print top results
    print_top_results(results_sorted, n=5)
    
    # Summary
    best = results_sorted.iloc[0]
    print("\n[6/6] Optimization Summary:")
    print("="*70)
    print(f"Total Combinations Tested: {len(results_df)}")
    print(f"Best Sharpe Ratio: {best['sharpe_ratio']:.4f}")
    print(f"Best Parameters:")
    print(f"  - Long Threshold:  {best['long_threshold']:.2f}%")
    print(f"  - Short Threshold: {best['short_threshold']:.2f}%")
    print(f"  - TX Cost:         {best['tx_cost']:.2f}%")
    print(f"\nOutputs:")
    print(f"  - {output_dir / 'threshold_grid_results.csv'}")
    print(f"  - {output_dir / 'threshold_best.json'}")
    print(f"  - {figures_dir / 'threshold_sharpe_surface.png'}")
    print(f"  - {figures_dir / 'threshold_mdd_surface.png'}")
    print("="*70)
    print("\n[SUCCESS] Threshold optimization completed!\n")


if __name__ == "__main__":
    main()
