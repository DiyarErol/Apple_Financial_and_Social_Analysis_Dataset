#!/usr/bin/env python3
"""
Backtest Simulator for Apple Stock Prediction Model
====================================================

Strategy:
- LONG if predicted_return > +0.5%
- SHORT if predicted_return < -0.5%
- FLAT otherwise
- Transaction cost: 0.1% per trade
- Starting capital: $100,000

Outputs:
- reports/final/backtest_results.csv (metrics summary + daily equity)
- reports/figures/equity_curve.png
- reports/figures/drawdown_chart.png
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports' / 'final'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
MODELS_DIR = BASE_DIR / 'models'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_predictions_data():
    """
    Load predictions and actuals.
    Priority:
    1. Use predictions_vs_actual.csv if exists
    2. Otherwise generate from model + dataset
    """
    
    print("[1/6] Loading predictions and actuals...")
    
    pred_csv = REPORTS_DIR / 'predictions_vs_actual.csv'
    
    if pred_csv.exists():
        print(f"[INFO] Using existing predictions: {pred_csv.name}")
        df = pd.read_csv(pred_csv)
        
        # Check required columns
        required = ['actual', 'predicted']
        if not all(col in df.columns for col in required):
            print("[WARN] predictions_vs_actual.csv missing required columns, regenerating...")
            return generate_predictions()
        
        # Convert prices to returns (percentage change)
        actual_prices = df['actual'].values
        predicted_prices = df['predicted'].values
        
        # Calculate returns: (price[t+1] - price[t]) / price[t] * 100
        actual_returns = np.zeros(len(actual_prices))
        predicted_returns = np.zeros(len(predicted_prices))
        
        for i in range(len(actual_prices) - 1):
            actual_returns[i] = ((actual_prices[i+1] - actual_prices[i]) / actual_prices[i]) * 100
            predicted_returns[i] = ((predicted_prices[i+1] - predicted_prices[i]) / predicted_prices[i]) * 100
        
        # Last row has no next price, set to 0
        actual_returns[-1] = 0
        predicted_returns[-1] = 0
        
        results_df = pd.DataFrame({
            'actual_return': actual_returns,
            'predicted_return': predicted_returns
        })
        
        # Try to get dates from main dataset
        main_df = pd.read_csv(DATA_DIR / 'apple_feature_enhanced.csv')
        if len(main_df) >= len(results_df):
            results_df['date'] = main_df['date'].iloc[:len(results_df)].values
            results_df['close'] = main_df['close'].iloc[:len(results_df)].values
        else:
            results_df['date'] = pd.date_range(start='2015-03-16', periods=len(results_df), freq='D')
            results_df['close'] = actual_prices  # use actual prices
        
        # Remove last row with 0 returns
        results_df = results_df.iloc[:-1]
        
        print(f"[OK] Loaded {len(results_df)} prediction samples (converted prices to returns)")
        return results_df
    
    else:
        print("[INFO] predictions_vs_actual.csv not found, generating from model...")
        return generate_predictions()


def generate_predictions():
    """Generate predictions from model and dataset."""
    
    # Load model
    model_path = MODELS_DIR / 'best_walkforward.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_cols = model_dict['feature_cols']
    
    print(f"[OK] Model loaded: {type(model).__name__}")
    
    # Load dataset
    csv_path = DATA_DIR / 'apple_feature_enhanced.csv'
    df = pd.read_csv(csv_path)
    
    # Engineer target (next day return)
    df['next_day_return'] = df['close'].pct_change().shift(-1) * 100
    df = df.dropna(subset=['next_day_return'])
    
    # Prepare features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Generate predictions
    y_pred = model.predict(X_scaled)
    
    # Build results dataframe
    results_df = pd.DataFrame({
        'date': df['date'].values,
        'close': df['close'].values,
        'actual_return': df['next_day_return'].values,
        'predicted_return': y_pred
    })
    
    print(f"[OK] Generated {len(results_df)} predictions")
    
    return results_df


def execute_backtest(df, initial_capital=100000, transaction_cost=0.001):
    """
    Execute backtest with long/short/flat strategy.
    
    Rules:
    - LONG: predicted_return > 0.5%
    - SHORT: predicted_return < -0.5%
    - FLAT: otherwise
    - Transaction cost: 0.1% per trade
    """
    
    print("[2/6] Executing backtest strategy...")
    
    LONG_THRESHOLD = 0.5
    SHORT_THRESHOLD = -0.5
    
    n = len(df)
    equity = np.zeros(n)
    position = np.zeros(n, dtype=int)
    trades = []
    
    equity[0] = initial_capital
    prev_position = 0
    
    for i in range(n):
        pred = df.loc[i, 'predicted_return']
        actual = df.loc[i, 'actual_return']
        
        # Skip if NaN
        if pd.isna(pred) or pd.isna(actual):
            if i > 0:
                equity[i] = equity[i-1]
            else:
                equity[i] = initial_capital
            position[i] = 0
            continue
        
        # Determine position
        if pred > LONG_THRESHOLD:
            new_position = 1
        elif pred < SHORT_THRESHOLD:
            new_position = -1
        else:
            new_position = 0
        
        # Start with previous equity
        if i > 0:
            equity[i] = equity[i-1]
        else:
            equity[i] = initial_capital
        
        # Apply transaction cost if position changed
        if new_position != prev_position:
            # Cost for closing old position + opening new position
            cost = transaction_cost
            if new_position != 0:
                cost += transaction_cost
            
            equity[i] *= (1 - cost)
            
            trades.append({
                'date': df.loc[i, 'date'],
                'from_position': prev_position,
                'to_position': new_position,
                'predicted': pred,
                'cost_pct': cost
            })
        
        # Apply strategy return
        strategy_return = new_position * (actual / 100)
        equity[i] *= (1 + strategy_return)
        
        position[i] = new_position
        prev_position = new_position
    
    df['equity'] = equity
    df['position'] = position
    
    print(f"[OK] Backtest executed: {len(trades)} trades, {len(df)} days")
    
    return df, trades


def compute_metrics(df, trades, initial_capital=100000):
    """Compute comprehensive backtest metrics."""
    
    print("[3/6] Computing performance metrics...")
    
    equity = df['equity'].values
    daily_returns = pd.Series(equity).pct_change().dropna()
    
    # Basic metrics
    final_equity = equity[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Annualized return (252 trading days)
    n_days = len(equity)
    n_years = n_days / 252
    annualized_return = (final_equity / initial_capital) ** (1 / n_years) - 1
    
    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (rf=0)
    if volatility > 0:
        sharpe = annualized_return / volatility
    else:
        sharpe = np.nan
    
    # Sortino Ratio (downside volatility only)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = annualized_return / downside_vol if downside_vol > 0 else np.nan
    else:
        sortino = np.nan
    
    # Maximum Drawdown
    cummax = pd.Series(equity).expanding().max()
    drawdown = (equity - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Win Rate (only for days with active positions)
    position_days = df[df['position'] != 0].copy()
    if len(position_days) > 0:
        position_returns = position_days['position'] * (position_days['actual_return'] / 100)
        wins = (position_returns > 0).sum()
        win_rate = wins / len(position_returns)
        
        # Average gain/loss
        winning_returns = position_returns[position_returns > 0]
        losing_returns = position_returns[position_returns < 0]
        
        avg_gain = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
    else:
        win_rate = 0
        avg_gain = 0
        avg_loss = 0
    
    # Average holding days
    if len(trades) > 1:
        holding_periods = []
        for i in range(len(trades) - 1):
            if trades[i]['to_position'] != 0:
                # Find next trade
                holding_periods.append(1)  # Simplified - daily granularity
        avg_holding_days = np.mean(holding_periods) if holding_periods else 0
    else:
        avg_holding_days = 0
    
    metrics = {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return': total_return * 100,  # percentage
        'annualized_return': annualized_return * 100,  # percentage
        'volatility_annual': volatility * 100,  # percentage
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown * 100,  # percentage
        'win_rate': win_rate * 100,  # percentage
        'avg_gain': avg_gain * 100,  # percentage
        'avg_loss': avg_loss * 100,  # percentage
        'trades_count': len(trades),
        'avg_holding_days': avg_holding_days,
        'trading_days': len(df)
    }
    
    print("[OK] Metrics computed")
    
    return metrics


def save_results(df, metrics, output_path):
    """Save backtest results to CSV."""
    
    print("[4/6] Saving results...")
    
    # Summary metrics
    summary_df = pd.DataFrame([{
        'metric': 'Initial Capital',
        'value': f"${metrics['initial_capital']:,.2f}"
    }, {
        'metric': 'Final Equity',
        'value': f"${metrics['final_equity']:,.2f}"
    }, {
        'metric': 'Total Return',
        'value': f"{metrics['total_return']:.2f}%"
    }, {
        'metric': 'Annualized Return',
        'value': f"{metrics['annualized_return']:.2f}%"
    }, {
        'metric': 'Volatility (Annual)',
        'value': f"{metrics['volatility_annual']:.2f}%"
    }, {
        'metric': 'Sharpe Ratio',
        'value': f"{metrics['sharpe_ratio']:.4f}" if not np.isnan(metrics['sharpe_ratio']) else "N/A"
    }, {
        'metric': 'Sortino Ratio',
        'value': f"{metrics['sortino_ratio']:.4f}" if not np.isnan(metrics['sortino_ratio']) else "N/A"
    }, {
        'metric': 'Max Drawdown',
        'value': f"{metrics['max_drawdown']:.2f}%"
    }, {
        'metric': 'Win Rate',
        'value': f"{metrics['win_rate']:.2f}%"
    }, {
        'metric': 'Avg Daily Gain',
        'value': f"{metrics['avg_gain']:.4f}%"
    }, {
        'metric': 'Avg Daily Loss',
        'value': f"{metrics['avg_loss']:.4f}%"
    }, {
        'metric': 'Total Trades',
        'value': str(metrics['trades_count'])
    }, {
        'metric': 'Trading Days',
        'value': str(metrics['trading_days'])
    }])
    
    summary_df.to_csv(output_path, index=False)
    
    # Also save daily equity curve
    equity_path = output_path.parent / 'equity_daily.csv'
    df[['date', 'equity', 'position']].to_csv(equity_path, index=False)
    
    print(f"[OK] Results saved: {output_path}")
    print(f"[OK] Daily equity saved: {equity_path}")


def plot_equity_curve(df, metrics, output_path):
    """Plot equity curve."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    equity = df['equity'].values
    
    # Equity curve
    ax1 = axes[0]
    ax1.plot(range(len(equity)), equity, linewidth=2, color='#2ecc71', label='Strategy Equity')
    ax1.axhline(y=metrics['initial_capital'], color='red', linestyle='--', 
                linewidth=1, alpha=0.7, label='Initial Capital')
    ax1.fill_between(range(len(equity)), metrics['initial_capital'], equity, 
                      alpha=0.2, color='#2ecc71')
    
    ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Backtest Equity Curve - Apple Stock Prediction Strategy', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Cumulative returns
    ax2 = axes[1]
    cumulative_returns = (equity / metrics['initial_capital'] - 1) * 100
    ax2.fill_between(range(len(equity)), 0, cumulative_returns, alpha=0.3, color='#2ecc71')
    ax2.plot(range(len(equity)), cumulative_returns, linewidth=1.5, 
             color='#2ecc71', label='Cumulative Return')
    
    ax2.set_xlabel('Trading Days', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Equity curve saved: {output_path}")


def plot_drawdown_chart(df, metrics, output_path):
    """Plot drawdown analysis."""
    
    equity = df['equity'].values
    cummax = pd.Series(equity).expanding().max().values
    drawdown = (equity - cummax) / cummax * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # Drawdown
    ax1 = axes[0]
    ax1.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.4, color='#e74c3c')
    ax1.plot(range(len(drawdown)), drawdown, linewidth=1, color='#c0392b')
    ax1.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Backtest Drawdown Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Running max vs current equity
    ax2 = axes[1]
    ax2.plot(range(len(cummax)), cummax, linewidth=2, 
             label='Running Maximum Equity', color='#3498db')
    ax2.plot(range(len(equity)), equity, linewidth=1.5, alpha=0.7, 
             label='Current Equity', color='#2ecc71')
    ax2.fill_between(range(len(equity)), equity, cummax, alpha=0.2, 
                      color='#e74c3c', label='Drawdown Amount')
    
    ax2.set_xlabel('Trading Days', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Drawdown chart saved: {output_path}")


def print_summary(metrics, df):
    """Print performance summary to console."""
    
    print("\n" + "="*70)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Initial Capital..................... ${metrics['initial_capital']:,.2f}")
    print(f"Final Equity........................ ${metrics['final_equity']:,.2f}")
    print(f"Total Return........................ {metrics['total_return']:.2f}%")
    print(f"Annualized Return................... {metrics['annualized_return']:.2f}%")
    print(f"Volatility (Annual)................. {metrics['volatility_annual']:.2f}%")
    
    if not np.isnan(metrics['sharpe_ratio']):
        print(f"Sharpe Ratio........................ {metrics['sharpe_ratio']:.4f}")
    else:
        print(f"Sharpe Ratio........................ N/A")
    
    if not np.isnan(metrics['sortino_ratio']):
        print(f"Sortino Ratio....................... {metrics['sortino_ratio']:.4f}")
    else:
        print(f"Sortino Ratio....................... N/A")
    
    print(f"Max Drawdown........................ {metrics['max_drawdown']:.2f}%")
    print(f"Win Rate............................ {metrics['win_rate']:.2f}%")
    print(f"Avg Daily Gain...................... {metrics['avg_gain']:.4f}%")
    print(f"Avg Daily Loss...................... {metrics['avg_loss']:.4f}%")
    print(f"Total Trades........................ {metrics['trades_count']}")
    print(f"Trading Days........................ {metrics['trading_days']}")
    print("="*70)
    
    if 'date' in df.columns:
        print(f"\nPeriod: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    
    print()


def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("BACKTEST SIMULATOR - Apple Stock Prediction Strategy")
    print("="*70 + "\n")
    
    try:
        # Load data
        df = load_predictions_data()
        
        # Execute backtest
        df, trades = execute_backtest(df)
        
        # Compute metrics
        metrics = compute_metrics(df, trades)
        
        # Save results
        results_path = REPORTS_DIR / 'backtest_results.csv'
        save_results(df, metrics, results_path)
        
        # Plot equity curve
        print("[5/6] Generating visualizations...")
        equity_plot_path = FIGURES_DIR / 'equity_curve.png'
        plot_equity_curve(df, metrics, equity_plot_path)
        
        # Plot drawdown
        drawdown_plot_path = FIGURES_DIR / 'drawdown_chart.png'
        plot_drawdown_chart(df, metrics, drawdown_plot_path)
        
        print("[6/6] Complete!")
        
        # Print summary
        print_summary(metrics, df)
        
        print("[SUCCESS] Backtest simulation completed!")
        print(f"\nOutputs:")
        print(f"  - {results_path}")
        print(f"  - {equity_plot_path}")
        print(f"  - {drawdown_plot_path}")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
