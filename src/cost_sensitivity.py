"""
cost_sensitivity.py
===================
Transaction cost sensitivity analysis for optimal backtest strategy.

Tests performance across transaction costs from 0.05% to 0.5%.
Uses best thresholds from grid search optimization.

Outputs:
- reports/final/cost_sensitivity.csv
- reports/figures/cost_vs_sharpe.png
- reports/figures/cost_vs_return.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """Get project root directory."""
    current = Path(__file__).resolve().parent
    if current.name == 'src':
        return current.parent
    return current


def load_best_thresholds():
    """Load best thresholds from optimization results."""
    root = get_project_root()
    json_path = root / 'reports' / 'final' / 'threshold_best.json'
    
    if not json_path.exists():
        print(f"[ERROR] Best thresholds file not found: {json_path}")
        print("[INFO] Please run optimize_thresholds.py first")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    params = data['best_parameters']
    print(f"[OK] Loaded best thresholds:")
    print(f"     Long:  {params['long_threshold']:.2f}%")
    print(f"     Short: {params['short_threshold']:.2f}%")
    
    return {
        'long_threshold': params['long_threshold'] / 100,  # Convert to decimal
        'short_threshold': params['short_threshold'] / 100
    }


def load_predictions_data():
    """
    Load predictions data from CSV.
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
    
    if not csv_path:
        print(f"[ERROR] predictions_vs_actual.csv not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Add date column if missing
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
    
    results_df = pd.DataFrame({
        'date': df['date'],
        'close': actual_prices,
        'actual_return': actual_returns,
        'predicted_return': predicted_returns
    })
    
    # Remove last row with 0 returns
    results_df = results_df.iloc[:-1].copy()
    
    print(f"[OK] Loaded {len(results_df)} prediction samples")
    return results_df


def execute_backtest(df, long_thr, short_thr, tx_cost, initial_capital=100000):
    """
    Execute backtest with given parameters.
    
    Returns dict with equity curve and trades.
    """
    n = len(df)
    equity = np.zeros(n)
    equity[0] = initial_capital
    
    positions = []
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
            if current_position != 0:
                cost += tx_cost
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
    """Compute performance metrics from equity curve."""
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
    
    # Sharpe ratio
    mean_return = np.mean(daily_returns)
    sharpe_ratio = (mean_return / volatility_daily) * np.sqrt(252) if volatility_daily > 0 else 0
    
    # Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
    sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    max_drawdown = np.min(drawdown)
    
    # Win rate
    winning_days = np.sum(daily_returns > 0)
    win_rate = (winning_days / len(daily_returns) * 100) if len(daily_returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility_annual': volatility_annual,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades_count': len(trades)
    }


def run_cost_sensitivity(df, thresholds, cost_range):
    """
    Run backtest across range of transaction costs.
    
    Args:
        df: Predictions DataFrame
        thresholds: Dict with long_threshold and short_threshold
        cost_range: Array of transaction costs to test
    
    Returns:
        DataFrame with results
    """
    print("\n[2/5] Running cost sensitivity analysis...")
    print(f"[INFO] Testing {len(cost_range)} transaction cost levels...")
    
    results = []
    
    for i, tx_cost in enumerate(cost_range, 1):
        # Run backtest
        backtest_result = execute_backtest(
            df,
            thresholds['long_threshold'],
            thresholds['short_threshold'],
            tx_cost
        )
        
        # Compute metrics
        metrics = compute_metrics(
            backtest_result['equity'],
            backtest_result['trades'],
            len(df)
        )
        
        # Store results
        result = {
            'tx_cost': tx_cost * 100,  # Convert to percentage
            **metrics
        }
        results.append(result)
        
        # Progress
        if i % 3 == 0 or i == len(cost_range):
            print(f"[PROGRESS] {i}/{len(cost_range)} cost levels tested")
    
    results_df = pd.DataFrame(results)
    print(f"[OK] Cost sensitivity analysis completed")
    
    return results_df


def save_results(results_df, output_dir):
    """Save cost sensitivity results to CSV."""
    print("\n[3/5] Saving results...")
    
    csv_path = output_dir / 'cost_sensitivity.csv'
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"[OK] Results saved: {csv_path}")
    
    return csv_path


def plot_results(results_df, figures_dir):
    """Create visualizations for cost sensitivity."""
    print("\n[4/5] Generating plots...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Cost vs Sharpe Ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results_df['tx_cost'], results_df['sharpe_ratio'], 
            marker='o', linewidth=2, markersize=8, color='#2ecc71', label='Sharpe Ratio')
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even (Sharpe=0)')
    
    # Add Sharpe=1 line (good performance threshold)
    ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Target (Sharpe=1)')
    
    ax.set_xlabel('Transaction Cost (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Transaction Cost Impact on Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate min and max points
    min_idx = results_df['sharpe_ratio'].idxmin()
    max_idx = results_df['sharpe_ratio'].idxmax()
    
    ax.annotate(f"Best: {results_df.loc[max_idx, 'sharpe_ratio']:.3f}\n@{results_df.loc[max_idx, 'tx_cost']:.2f}%",
                xy=(results_df.loc[max_idx, 'tx_cost'], results_df.loc[max_idx, 'sharpe_ratio']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                fontsize=9)
    
    plt.tight_layout()
    sharpe_path = figures_dir / 'cost_vs_sharpe.png'
    plt.savefig(sharpe_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Sharpe plot saved: {sharpe_path}")
    
    # Plot 2: Cost vs Return and Max Drawdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Total Return
    ax1.plot(results_df['tx_cost'], results_df['total_return'], 
             marker='s', linewidth=2, markersize=8, color='#3498db', label='Total Return')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_xlabel('Transaction Cost (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Transaction Cost Impact on Total Return', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Annotate best return
    max_return_idx = results_df['total_return'].idxmax()
    ax1.annotate(f"Best: {results_df.loc[max_return_idx, 'total_return']:.2f}%\n@{results_df.loc[max_return_idx, 'tx_cost']:.2f}%",
                 xy=(results_df.loc[max_return_idx, 'tx_cost'], results_df.loc[max_return_idx, 'total_return']),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                 fontsize=9)
    
    # Max Drawdown
    ax2.plot(results_df['tx_cost'], results_df['max_drawdown'], 
             marker='^', linewidth=2, markersize=8, color='#e74c3c', label='Max Drawdown')
    ax2.set_xlabel('Transaction Cost (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Transaction Cost Impact on Max Drawdown', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Annotate best (least negative) drawdown
    best_dd_idx = results_df['max_drawdown'].idxmax()
    ax2.annotate(f"Best: {results_df.loc[best_dd_idx, 'max_drawdown']:.2f}%\n@{results_df.loc[best_dd_idx, 'tx_cost']:.2f}%",
                 xy=(results_df.loc[best_dd_idx, 'tx_cost'], results_df.loc[best_dd_idx, 'max_drawdown']),
                 xytext=(10, -20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7),
                 fontsize=9)
    
    plt.tight_layout()
    return_path = figures_dir / 'cost_vs_return.png'
    plt.savefig(return_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Return/Drawdown plot saved: {return_path}")


def find_breakeven_cost(results_df, target_sharpe=1.0):
    """
    Find transaction cost where Sharpe ratio reaches target.
    
    Returns:
        Dict with breakeven info or None if not achievable
    """
    print("\n[5/5] Analyzing break-even cost...")
    
    # Check if target is achievable
    max_sharpe = results_df['sharpe_ratio'].max()
    
    if max_sharpe < target_sharpe:
        print(f"[INFO] Target Sharpe ({target_sharpe}) not achievable with current strategy")
        print(f"[INFO] Maximum Sharpe achieved: {max_sharpe:.4f}")
        
        # Find cost that gives Sharpe closest to zero (true break-even)
        zero_idx = (results_df['sharpe_ratio'] - 0).abs().idxmin()
        zero_cost = results_df.loc[zero_idx, 'tx_cost']
        zero_sharpe = results_df.loc[zero_idx, 'sharpe_ratio']
        
        print(f"[INFO] Closest to break-even (Sharpe≈0): {zero_cost:.3f}% cost → Sharpe={zero_sharpe:.4f}")
        
        return {
            'target_sharpe': target_sharpe,
            'achievable': False,
            'max_sharpe': max_sharpe,
            'zero_breakeven_cost': zero_cost,
            'zero_breakeven_sharpe': zero_sharpe
        }
    
    # Interpolate to find exact breakeven cost
    # Find two points that bracket the target
    above = results_df[results_df['sharpe_ratio'] >= target_sharpe]
    
    if len(above) > 0:
        breakeven_row = above.iloc[-1]  # Highest cost that still meets target
        breakeven_cost = breakeven_row['tx_cost']
        breakeven_sharpe = breakeven_row['sharpe_ratio']
        
        print(f"[OK] Break-even cost for Sharpe={target_sharpe}: {breakeven_cost:.3f}%")
        print(f"     Actual Sharpe at this cost: {breakeven_sharpe:.4f}")
        
        return {
            'target_sharpe': target_sharpe,
            'achievable': True,
            'breakeven_cost': breakeven_cost,
            'breakeven_sharpe': breakeven_sharpe
        }
    
    return None


def print_summary(results_df, thresholds):
    """Print comprehensive summary."""
    print("\n" + "="*70)
    print("COST SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n[Strategy Parameters]")
    print(f"  Long Threshold:  {thresholds['long_threshold']*100:.2f}%")
    print(f"  Short Threshold: {thresholds['short_threshold']*100:.2f}%")
    
    print(f"\n[Cost Range Tested]")
    print(f"  Min Cost: {results_df['tx_cost'].min():.3f}%")
    print(f"  Max Cost: {results_df['tx_cost'].max():.3f}%")
    print(f"  Samples:  {len(results_df)}")
    
    print(f"\n[Best Performance (Highest Sharpe)]")
    best_idx = results_df['sharpe_ratio'].idxmax()
    best = results_df.loc[best_idx]
    print(f"  TX Cost:       {best['tx_cost']:.3f}%")
    print(f"  Sharpe Ratio:  {best['sharpe_ratio']:.4f}")
    print(f"  Total Return:  {best['total_return']:.2f}%")
    print(f"  Max Drawdown:  {best['max_drawdown']:.2f}%")
    print(f"  Win Rate:      {best['win_rate']:.2f}%")
    
    print(f"\n[Worst Performance (Lowest Sharpe)]")
    worst_idx = results_df['sharpe_ratio'].idxmin()
    worst = results_df.loc[worst_idx]
    print(f"  TX Cost:       {worst['tx_cost']:.3f}%")
    print(f"  Sharpe Ratio:  {worst['sharpe_ratio']:.4f}")
    print(f"  Total Return:  {worst['total_return']:.2f}%")
    print(f"  Max Drawdown:  {worst['max_drawdown']:.2f}%")
    
    print(f"\n[Cost Impact]")
    sharpe_range = results_df['sharpe_ratio'].max() - results_df['sharpe_ratio'].min()
    return_range = results_df['total_return'].max() - results_df['total_return'].min()
    print(f"  Sharpe Range:  {sharpe_range:.4f}")
    print(f"  Return Range:  {return_range:.2f}%")
    print(f"  Sensitivity:   {return_range / (results_df['tx_cost'].max() - results_df['tx_cost'].min()):.2f}% return per 1% cost")
    
    print("\n" + "="*70)


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("COST SENSITIVITY ANALYSIS - Transaction Cost Impact")
    print("="*70 + "\n")
    
    # Setup paths
    root = get_project_root()
    output_dir = root / 'reports' / 'final'
    figures_dir = root / 'reports' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load best thresholds
    print("[1/5] Loading best thresholds and predictions...")
    thresholds = load_best_thresholds()
    
    if thresholds is None:
        sys.exit(1)
    
    # Load predictions
    df = load_predictions_data()
    
    if df is None or len(df) == 0:
        print("\n[ERROR] Failed to load predictions data")
        sys.exit(1)
    
    # Define cost range: 0.05% to 0.5%
    cost_range = np.linspace(0.0005, 0.005, 10)
    
    # Run sensitivity analysis
    results_df = run_cost_sensitivity(df, thresholds, cost_range)
    
    # Save results
    save_results(results_df, output_dir)
    
    # Plot results
    plot_results(results_df, figures_dir)
    
    # Find break-even
    breakeven_info = find_breakeven_cost(results_df, target_sharpe=1.0)
    
    # Print summary
    print_summary(results_df, thresholds)
    
    print(f"\n[SUCCESS] Cost sensitivity analysis completed!\n")


if __name__ == "__main__":
    main()
