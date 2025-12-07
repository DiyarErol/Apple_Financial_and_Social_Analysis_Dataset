"""
Interaction Features Engineering Script
Apple Financial and Social Analysis Dataset

Adds interaction and regime features on top of apple_with_sentiment.csv and
saves apple_with_interactions.csv plus a momentum regimes plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


REQUIRED_COLS = {
    'date', 'close', 'rsi_14', 'macd', 'macd_signal', 'volume', 'vix',
    'fed_rate', 'gold', 'daily_return'
}


# ============================================================================
# Helpers
# ============================================================================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("INTERACTION FEATURES PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("=" * 80)
    print("\n[1/5] Loading dataset...")

    # Try to load apple_with_sentiment.csv first, fallback to apple_with_events.csv
    if not csv_path.exists():
        fallback_path = csv_path.parent / 'apple_with_events.csv'
        if fallback_path.exists():
            print(f"⚠ {csv_path.name} not found, using fallback: {fallback_path.name}")
            csv_path = fallback_path
        else:
            raise SystemExit(f"Cannot find {csv_path} or fallback {fallback_path}")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def compute_macd_crossover(macd_diff: pd.Series) -> pd.Series:
    # sign of MACD minus signal
    sign_series = np.sign(macd_diff)
    crossover = sign_series.diff().fillna(0).clip(-1, 1)
    return crossover


def compute_crossover_days_ago(crossover: pd.Series) -> pd.Series:
    counter = []
    last_cross = None
    for i, val in enumerate(crossover.values):
        if val != 0:
            last_cross = i
            counter.append(0)
        else:
            if last_cross is None:
                counter.append(np.nan)
            else:
                counter.append(i - last_cross)
    return pd.Series(counter, index=crossover.index)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Adding interaction and regime features...")
    out = df.copy()

    # RSI divergence
    out['rsi_divergence'] = out['close'].pct_change() - out['rsi_14'].pct_change()

    # RSI flags
    out['is_rsi_oversold'] = (out['rsi_14'] < 30).astype(int)
    out['is_rsi_overbought'] = (out['rsi_14'] > 70).astype(int)

    # MACD crossover
    macd_diff = out['macd'] - out['macd_signal']
    out['macd_crossover'] = compute_macd_crossover(macd_diff)
    out['macd_crossover_days_ago'] = compute_crossover_days_ago(out['macd_crossover'])

    # Volume features
    out['volume_ma_20'] = out['volume'].rolling(20, min_periods=1).mean()
    out['volume_spike'] = (out['volume'] > 2 * out['volume_ma_20']).astype(int)
    out['volume_ratio'] = out['volume'] / out['volume_ma_20']

    # Interaction terms
    out['fed_rate_x_vix'] = out['fed_rate'] * out['vix']
    out['volume_x_volatility'] = out['volume'] * out['vix']
    trend_delta = out['close'].pct_change()
    out['trend_delta_x_daily_return'] = trend_delta * out['daily_return']
    out['rsi_x_macd'] = out['rsi_14'] * out['macd']
    out['vix_x_gold'] = out['vix'] * out['gold']

    # Volatility regime
    out['volatility_regime'] = np.select(
        [out['vix'] > 25, out['vix'] > 15], [2, 1], default=0
    ).astype(int)

    print("✓ Interaction features added")
    return out


def save_dataset(df: pd.DataFrame, output_path: Path):
    print("\n[3/5] Saving dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(df):,} rows, {len(df.columns)} cols)")


def plot_momentum_regimes(df: pd.DataFrame, output_path: Path):
    print("\n[4/5] Plotting momentum regimes (RSI + MACD)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Price and MACD diff
    ax1.plot(df['date'], df['close'], color='#1f77b4', label='Close', linewidth=1.4)
    ax1.set_ylabel('Close ($)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    macd_diff = df['macd'] - df['macd_signal']
    ax2.plot(df['date'], macd_diff, color='#ff7f0e', label='MACD - Signal', linewidth=1.1)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Highlight RSI overbought/oversold regimes
    oversold = df['is_rsi_oversold'] == 1
    overbought = df['is_rsi_overbought'] == 1
    ax2.fill_between(df['date'], macd_diff, where=oversold, color='green', alpha=0.15, label='RSI Oversold')
    ax2.fill_between(df['date'], macd_diff, where=overbought, color='red', alpha=0.12, label='RSI Overbought')

    # Mark crossovers
    cross_idx = df.index[df['macd_crossover'] != 0]
    ax2.scatter(df.loc[cross_idx, 'date'], macd_diff.loc[cross_idx], color='black', s=10, label='MACD Crossover')

    ax2.set_ylabel('MACD - Signal', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {output_path}")


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data' / 'processed'
    reports_dir = script_dir / 'reports' / 'figures'

    input_csv = data_dir / 'apple_with_sentiment.csv'
    output_csv = data_dir / 'apple_with_interactions.csv'
    output_fig = reports_dir / 'momentum_regimes.png'

    df = load_dataset(input_csv)
    df_feat = add_interaction_features(df)
    save_dataset(df_feat, output_csv)
    plot_momentum_regimes(df_feat, output_fig)

    print("\n" + "=" * 80)
    print("INTERACTION FEATURES PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n✓ Output CSV: {output_csv}")
    print(f"✓ Output Figure: {output_fig}")


if __name__ == '__main__':
    main()
