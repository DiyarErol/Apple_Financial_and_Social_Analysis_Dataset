"""
China Supply Chain & Global Macro Integration Script
Apple Financial and Social Analysis Dataset

Fetches economic indicators (China retail sales, CNY/USD, semiconductors, shipping, copper)
and merges them with the main dataset.

Data sources:
- China retail sales: FRED CHNCSTOSM (monthly → forward-fill daily)
- CNY/USD rate: Yahoo Finance CNY=X
- Semiconductor index: Yahoo Finance ^SOX
- Shipping cost index: Yahoo Finance ^BDI (Baltic Dry)
- Copper price: Yahoo Finance HG=F
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit("yfinance is required. Please install with: pip install yfinance") from exc

try:
    import pandas_datareader as pdr
except ImportError as exc:
    raise SystemExit("pandas_datareader is required. Please install with: pip install pandas_datareader") from exc


# ============================================================================
# Loaders
# ============================================================================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("CHINA SUPPLY CHAIN & GLOBAL MACRO PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("=" * 80)
    print("\n[1/7] Loading base dataset...")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    print(f"✓ Loaded {len(df):,} rows")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def fetch_china_retail_sales():
    """Fetch China retail sales from FRED (monthly) and forward-fill to daily."""
    print("\n[2/7] Fetching China retail sales (FRED CHNCSTOSM)...")
    try:
        df = pdr.get_data_fred('CHNCSTOSM', start='2015-01-01')
        df = df.rename(columns={'CHNCSTOSM': 'china_retail_sales'})
        df.index = pd.to_datetime(df.index).normalize()
        df = df.asfreq('D').fillna(method='ffill')  # forward-fill daily
        print(f"✓ China retail sales: {len(df):,} days")
        return df
    except Exception as exc:
        print(f"⚠ Error fetching CHNCSTOSM: {exc}")
        return pd.DataFrame()


def fetch_yfinance_series(ticker: str, label: str, start: str = '2015-01-01') -> pd.DataFrame:
    """Fetch daily series from Yahoo Finance."""
    print(f"\n  Fetching {label} ({ticker})...")
    try:
        df = yf.download(ticker, start=start, progress=False)['Adj Close'].to_frame()
        df = df.rename(columns={'Adj Close': label})
        df.index = pd.to_datetime(df.index).normalize()
        df = df.fillna(method='ffill')  # forward-fill missing days
        print(f"    ✓ {label}: {len(df):,} days")
        return df
    except Exception as exc:
        print(f"    ⚠ Error fetching {ticker}: {exc}")
        return pd.DataFrame()


def fetch_global_macro_data():
    """Fetch all global macro series."""
    print("\n[3/7] Fetching global macro indicators from Yahoo Finance...")

    dfs = {}

    # CNY/USD rate
    cny_usd = fetch_yfinance_series('CNY=X', 'yuan_usd_rate')
    if not cny_usd.empty:
        dfs['cny_usd'] = cny_usd

    # Semiconductor index (SOX)
    sox = fetch_yfinance_series('^SOX', 'semiconductor_index')
    if not sox.empty:
        dfs['sox'] = sox

    # Baltic Dry Index (BDI) - shipping costs
    bdi = fetch_yfinance_series('^BDI', 'shipping_cost_index')
    if not bdi.empty:
        dfs['bdi'] = bdi

    # Copper price (HG=F)
    copper = fetch_yfinance_series('HG=F', 'copper_price')
    if not copper.empty:
        dfs['copper'] = copper

    return dfs


def merge_all_series(df_base: pd.DataFrame, fred_df: pd.DataFrame, yf_dfs: dict) -> pd.DataFrame:
    """Merge FRED and Yahoo Finance series into base dataset."""
    print("\n[4/7] Merging all series on date...")

    result = df_base.copy()

    # Merge FRED data
    if not fred_df.empty:
        result = result.merge(fred_df, left_on='date', right_index=True, how='left')

    # Merge Yahoo Finance data
    for label, df in yf_dfs.items():
        result = result.merge(df, left_on='date', right_index=True, how='left')

    # Forward-fill any remaining NaNs within the dataset range
    result = result.fillna(method='ffill')

    print(f"✓ Merged dataset: {len(result):,} rows, {len(result.columns)} columns")
    return result


def save_dataset(df: pd.DataFrame, output_path: Path):
    """Save merged dataset."""
    print("\n[5/7] Saving enhanced dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")


def plot_global_macro_overlay(df: pd.DataFrame, output_path: Path):
    """Plot normalized close vs global macro indicators."""
    print("\n[6/7] Plotting global macro overlay...")

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Normalize close price to [0, 100] for comparison
    close_norm = 100 * (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())

    # Plot 1: Close price
    axes[0].plot(df['date'], close_norm, color='#1f77b4', label='Close (normalized)', linewidth=1.5)
    axes[0].set_ylabel('Close ($)', fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Semiconductor index (SOX)
    if 'semiconductor_index' in df.columns:
        sox_norm = 100 * (df['semiconductor_index'] - df['semiconductor_index'].min()) / \
                   (df['semiconductor_index'].max() - df['semiconductor_index'].min())
        axes[1].plot(df['date'], sox_norm, color='#ff7f0e', label='Semiconductor Index (SOX)', linewidth=1.2)
        axes[1].set_ylabel('SOX (normalized)', fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'SOX data unavailable', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_ylabel('SOX', fontweight='bold')

    # Plot 3: CNY/USD and BDI
    ax3 = axes[2]
    if 'yuan_usd_rate' in df.columns:
        cny_norm = 100 * (df['yuan_usd_rate'] - df['yuan_usd_rate'].min()) / \
                   (df['yuan_usd_rate'].max() - df['yuan_usd_rate'].min())
        ax3.plot(df['date'], cny_norm, color='#2ca02c', label='CNY/USD Rate', linewidth=1.2)
    ax3_b = ax3.twinx()
    if 'shipping_cost_index' in df.columns:
        bdi_norm = 100 * (df['shipping_cost_index'] - df['shipping_cost_index'].min()) / \
                   (df['shipping_cost_index'].max() - df['shipping_cost_index'].min())
        ax3_b.plot(df['date'], bdi_norm, color='#d62728', label='Shipping Cost (BDI)', linewidth=1.2, linestyle='--')
        ax3_b.set_ylabel('BDI (normalized)', fontweight='bold', color='#d62728')
    ax3.set_ylabel('CNY/USD (normalized)', fontweight='bold', color='#2ca02c')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='#2ca02c')
    if 'shipping_cost_index' in df.columns:
        ax3_b.tick_params(axis='y', labelcolor='#d62728')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot 4: Copper price
    if 'copper_price' in df.columns:
        copper_norm = 100 * (df['copper_price'] - df['copper_price'].min()) / \
                      (df['copper_price'].max() - df['copper_price'].min())
        axes[3].plot(df['date'], copper_norm, color='#9467bd', label='Copper Price', linewidth=1.2)
        axes[3].set_ylabel('Copper (normalized)', fontweight='bold')
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'Copper data unavailable', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_ylabel('Copper', fontweight='bold')

    axes[3].set_xlabel('Date', fontweight='bold')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data' / 'processed'
    reports_dir = script_dir / 'reports' / 'figures'

    input_csv = data_dir / 'apple_with_interactions.csv'
    output_csv = data_dir / 'apple_feature_enhanced.csv'
    output_fig = reports_dir / 'global_macro_overlay.png'

    # Load base dataset
    df = load_dataset(input_csv)

    # Fetch FRED data
    df_fred = fetch_china_retail_sales()

    # Fetch Yahoo Finance data
    yf_dfs = fetch_global_macro_data()

    # Merge
    df_merged = merge_all_series(df, df_fred, yf_dfs)

    # Save
    save_dataset(df_merged, output_csv)

    # Plot
    plot_global_macro_overlay(df_merged, output_fig)

    # Summary
    print("\n" + "=" * 80)
    print("CHINA SUPPLY CHAIN & GLOBAL MACRO PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n✓ Output CSV: {output_csv}")
    print(f"✓ Output Figure: {output_fig}")
    print(f"✓ New features: china_retail_sales, yuan_usd_rate, semiconductor_index, shipping_cost_index, copper_price")


if __name__ == '__main__':
    main()
