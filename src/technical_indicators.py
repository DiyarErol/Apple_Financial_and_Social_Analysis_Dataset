"""
Technical Indicators Script for Apple Financial and Social Analysis Dataset.
Calculates common technical indicators using pandas only (no TA-lib).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def load_macro_dataset(csv_path):
    """
    Load macroeconomic dataset.
    
    Parameters:
    -----------
    csv_path : str
        Path to the macro dataset CSV
    
    Returns:
    --------
    pd.DataFrame
        Macro dataset with date converted to datetime
    """
    print("Loading macroeconomic dataset...")
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_rsi(df, column='close', period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    column : str
        Column name for price data
    period : int
        RSI period (default 14)
    
    Returns:
    --------
    pd.Series
        RSI values
    """
    print(f"Calculating RSI ({period})...")
    try:
        delta = df[column].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        print(f"✓ RSI calculated (period={period})")
        print(f"  Range: {rsi.min():.2f} to {rsi.max():.2f}")
        
        return rsi
    except Exception as e:
        print(f"✗ Error calculating RSI: {e}")
        return pd.Series(np.nan, index=df.index)


def calculate_macd(df, column='close', fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    column : str
        Column name for price data
    fast : int
        Fast EMA period (default 12)
    slow : int
        Slow EMA period (default 26)
    signal : int
        Signal line EMA period (default 9)
    
    Returns:
    --------
    dict
        Dictionary with MACD, signal, and histogram
    """
    print(f"Calculating MACD ({fast}-{slow} EMA, {signal} signal)...")
    try:
        # Calculate EMAs
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        # MACD histogram
        macd_hist = macd - macd_signal
        
        print(f"✓ MACD calculated")
        print(f"  MACD range: {macd.min():.4f} to {macd.max():.4f}")
        print(f"  Histogram range: {macd_hist.min():.4f} to {macd_hist.max():.4f}")
        
        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist
        }
    except Exception as e:
        print(f"✗ Error calculating MACD: {e}")
        return {
            'macd': pd.Series(np.nan, index=df.index),
            'macd_signal': pd.Series(np.nan, index=df.index),
            'macd_hist': pd.Series(np.nan, index=df.index)
        }


def calculate_bollinger_bands(df, column='close', period=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    column : str
        Column name for price data
    period : int
        Moving average period (default 20)
    num_std : float
        Number of standard deviations (default 2)
    
    Returns:
    --------
    dict
        Dictionary with middle, upper, and lower bands
    """
    print(f"Calculating Bollinger Bands ({period}, {num_std} std)...")
    try:
        # Middle band (SMA)
        sma = df[column].rolling(window=period).mean()
        
        # Standard deviation
        std = df[column].rolling(window=period).std()
        
        # Upper and lower bands
        bb_upper = sma + (std * num_std)
        bb_lower = sma - (std * num_std)
        
        print(f"✓ Bollinger Bands calculated")
        print(f"  SMA range: {sma.min():.2f} to {sma.max():.2f}")
        print(f"  Band width: {(bb_upper - bb_lower).mean():.2f} (avg)")
        
        return {
            'bb_mid': sma,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower
        }
    except Exception as e:
        print(f"✗ Error calculating Bollinger Bands: {e}")
        return {
            'bb_mid': pd.Series(np.nan, index=df.index),
            'bb_upper': pd.Series(np.nan, index=df.index),
            'bb_lower': pd.Series(np.nan, index=df.index)
        }


def calculate_atr(df, high='high', low='low', close='close', period=14):
    """
    Calculate Average True Range (ATR).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    high : str
        High column name
    low : str
        Low column name
    close : str
        Close column name
    period : int
        ATR period (default 14)
    
    Returns:
    --------
    pd.Series
        ATR values
    """
    print(f"Calculating ATR ({period})...")
    try:
        # True range components
        hl = df[high] - df[low]
        hc = np.abs(df[high] - df[close].shift(1))
        lc = np.abs(df[low] - df[close].shift(1))
        
        # True range
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        # ATR (rolling average of true range)
        atr = tr.rolling(window=period).mean()
        
        print(f"✓ ATR calculated (period={period})")
        print(f"  Range: {atr.min():.2f} to {atr.max():.2f}")
        
        return atr
    except Exception as e:
        print(f"✗ Error calculating ATR: {e}")
        return pd.Series(np.nan, index=df.index)


def calculate_obv(df, close='close', volume='volume'):
    """
    Calculate On-Balance Volume (OBV).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with close and volume data
    close : str
        Close column name
    volume : str
        Volume column name
    
    Returns:
    --------
    pd.Series
        OBV values
    """
    print("Calculating On-Balance Volume (OBV)...")
    try:
        # Price changes direction
        obv = pd.Series(np.nan, index=df.index)
        obv.iloc[0] = df[volume].iloc[0]
        
        for i in range(1, len(df)):
            if df[close].iloc[i] > df[close].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df[volume].iloc[i]
            elif df[close].iloc[i] < df[close].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df[volume].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        print(f"✓ OBV calculated")
        print(f"  Range: {obv.min():.0f} to {obv.max():.0f}")
        
        return obv
    except Exception as e:
        print(f"✗ Error calculating OBV: {e}")
        return pd.Series(np.nan, index=df.index)


def calculate_vwap(df, high='high', low='low', close='close', volume='volume', period=20):
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    high : str
        High column name
    low : str
        Low column name
    close : str
        Close column name
    volume : str
        Volume column name
    period : int
        VWAP period (default 20)
    
    Returns:
    --------
    pd.Series
        VWAP values
    """
    print(f"Calculating VWAP ({period})...")
    try:
        # Typical price
        tp = (df[high] + df[low] + df[close]) / 3
        
        # Cumulative TP * Volume
        tp_vol = (tp * df[volume]).rolling(window=period).sum()
        
        # Cumulative Volume
        cum_vol = df[volume].rolling(window=period).sum()
        
        # VWAP
        vwap = tp_vol / cum_vol.replace(0, np.nan)
        
        print(f"✓ VWAP calculated (period={period})")
        print(f"  Range: {vwap.min():.2f} to {vwap.max():.2f}")
        
        return vwap
    except Exception as e:
        print(f"✗ Error calculating VWAP: {e}")
        return pd.Series(np.nan, index=df.index)


def add_technical_indicators(df):
    """
    Add all technical indicators to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with technical indicators added
    """
    print("\n" + "="*80)
    print("CALCULATING TECHNICAL INDICATORS")
    print("="*80 + "\n")
    
    df_ti = df.copy()
    
    # RSI
    df_ti['rsi_14'] = calculate_rsi(df_ti, column='close', period=14)
    
    # MACD
    print()
    macd_dict = calculate_macd(df_ti, column='close', fast=12, slow=26, signal=9)
    df_ti['macd'] = macd_dict['macd']
    df_ti['macd_signal'] = macd_dict['macd_signal']
    df_ti['macd_hist'] = macd_dict['macd_hist']
    
    # Bollinger Bands
    print()
    bb_dict = calculate_bollinger_bands(df_ti, column='close', period=20, num_std=2)
    df_ti['bb_mid_20'] = bb_dict['bb_mid']
    df_ti['bb_upper_20'] = bb_dict['bb_upper']
    df_ti['bb_lower_20'] = bb_dict['bb_lower']
    
    # ATR
    print()
    df_ti['atr_14'] = calculate_atr(df_ti, high='high', low='low', close='close', period=14)
    
    # OBV
    print()
    df_ti['obv'] = calculate_obv(df_ti, close='close', volume='volume')
    
    # VWAP
    print()
    df_ti['vwap_20'] = calculate_vwap(df_ti, high='high', low='low', close='close', 
                                      volume='volume', period=20)
    
    return df_ti


def clean_data(df):
    """
    Remove rows with NaN values introduced by rolling windows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with indicators
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("\nCleaning data...")
    try:
        initial_rows = len(df)
        
        # Drop NaN rows
        df_clean = df.dropna()
        
        rows_dropped = initial_rows - len(df_clean)
        
        print(f"✓ Data cleaned")
        print(f"  Initial rows: {initial_rows}")
        print(f"  Rows dropped: {rows_dropped}")
        print(f"  Final rows: {len(df_clean)}")
        print(f"  Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
        
        return df_clean
    except Exception as e:
        print(f"✗ Error cleaning data: {e}")
        return df


def save_dataset(df, output_path):
    """
    Save dataset to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    output_path : str
        Path to save CSV
    """
    print("Saving dataset...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset saved to {output_path}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")


def plot_indicators_overview(df, output_dir):
    """
    Generate overview plot with close price, RSI, and MACD histogram.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with indicators
    output_dir : str
        Output directory for the plot
    """
    print("Generating indicators overview plot...")
    try:
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        # Close price
        axes[0].plot(df['date'], df['close'], color='#1f77b4', linewidth=2)
        axes[0].fill_between(df['date'], df['close'], alpha=0.3, color='#1f77b4')
        axes[0].set_ylabel('Close Price (USD)', fontsize=11, fontweight='bold')
        axes[0].set_title('Apple Stock Price with Technical Indicators Overview', 
                         fontsize=14, fontweight='bold', pad=15)
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        axes[1].plot(df['date'], df['rsi_14'], color='#ff7f0e', linewidth=2, label='RSI(14)')
        axes[1].axhline(y=70, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Overbought (70)')
        axes[1].axhline(y=30, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Oversold (30)')
        axes[1].fill_between(df['date'], 30, 70, alpha=0.1, color='gray')
        axes[1].set_ylabel('RSI', fontsize=11, fontweight='bold')
        axes[1].set_ylim([0, 100])
        axes[1].legend(loc='upper left', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # MACD Histogram
        colors = ['green' if x > 0 else 'red' for x in df['macd_hist']]
        axes[2].bar(df['date'], df['macd_hist'], color=colors, alpha=0.6, width=1, label='MACD Histogram')
        axes[2].plot(df['date'], df['macd'], color='#2ca02c', linewidth=2, label='MACD')
        axes[2].plot(df['date'], df['macd_signal'], color='#d62728', linewidth=2, label='Signal Line')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[2].set_ylabel('MACD', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
        axes[2].legend(loc='upper left', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'indicators_overview.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Indicators overview plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating indicators overview plot: {e}")


def plot_bollinger_bands(df, output_dir):
    """
    Generate Bollinger Bands plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with indicators
    output_dir : str
        Output directory for the plot
    """
    print("Generating Bollinger Bands plot...")
    try:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Bollinger Bands
        ax.fill_between(df['date'], df['bb_upper_20'], df['bb_lower_20'], 
                       alpha=0.2, color='#1f77b4', label='Bollinger Bands (20, 2 std)')
        ax.plot(df['date'], df['bb_upper_20'], color='#1f77b4', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Upper Band')
        ax.plot(df['date'], df['bb_mid_20'], color='#ff7f0e', linewidth=2, label='Middle Band (SMA 20)')
        ax.plot(df['date'], df['bb_lower_20'], color='#1f77b4', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Lower Band')
        
        # Close price
        ax.plot(df['date'], df['close'], color='#2ca02c', linewidth=2.5, label='Close Price', zorder=5)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Apple Stock Price with Bollinger Bands (20, 2 std)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'bollinger_example.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Bollinger Bands plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating Bollinger Bands plot: {e}")


def print_indicator_statistics(df):
    """
    Print statistics for all technical indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with indicators
    """
    print("\n" + "="*80)
    print("TECHNICAL INDICATOR STATISTICS")
    print("="*80 + "\n")
    
    indicators = ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_mid_20', 
                 'bb_upper_20', 'bb_lower_20', 'atr_14', 'obv', 'vwap_20']
    
    for ind in indicators:
        if ind in df.columns:
            print(f"{ind}:")
            print(f"  Mean:     {df[ind].mean():.4f}")
            print(f"  Median:   {df[ind].median():.4f}")
            print(f"  Std:      {df[ind].std():.4f}")
            print(f"  Min:      {df[ind].min():.4f}")
            print(f"  Max:      {df[ind].max():.4f}")
            print()


def main():
    """
    Main execution function:
    1. Load macro dataset
    2. Calculate technical indicators
    3. Clean data (remove NaN)
    4. Save to CSV
    5. Generate visualizations
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL ANALYSIS - TECHNICAL INDICATORS CALCULATION")
    print("="*80 + "\n")
    
    # Define paths
    input_csv = 'data/processed/apple_macro_dataset.csv'
    output_csv = 'data/processed/apple_with_ti.csv'
    figures_dir = 'reports/figures'
    
    # Step 1: Load dataset
    print("[1/5] Loading macroeconomic dataset...")
    df = load_macro_dataset(input_csv)
    if df is None:
        print("✗ Failed to load dataset. Exiting.")
        return False
    
    # Step 2: Add technical indicators
    print("\n[2/5] Adding technical indicators...")
    df_with_ti = add_technical_indicators(df)
    
    # Step 3: Clean data
    print("\n[3/5] Cleaning data...")
    df_clean = clean_data(df_with_ti)
    
    # Step 4: Save dataset
    print("\n[4/5] Saving dataset...")
    save_dataset(df_clean, output_csv)
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    plot_indicators_overview(df_clean, figures_dir)
    plot_bollinger_bands(df_clean, figures_dir)
    
    # Print statistics
    print_indicator_statistics(df_clean)
    
    # Final summary
    print("="*80)
    print("TECHNICAL INDICATORS CALCULATION COMPLETE")
    print("="*80)
    
    print(f"\n✓ Output Files:")
    print(f"  • {output_csv}")
    print(f"  • {figures_dir}/indicators_overview.png")
    print(f"  • {figures_dir}/bollinger_example.png")
    
    print(f"\n✓ Technical Indicators Added:")
    print(f"  • RSI (14-period)")
    print(f"  • MACD (12-26 EMA, 9 signal)")
    print(f"  • Bollinger Bands (20, 2 std)")
    print(f"  • ATR (14-period)")
    print(f"  • OBV (On-Balance Volume)")
    print(f"  • VWAP (20-day rolling)")
    
    print(f"\n✓ Data Quality:")
    print(f"  • Records: {len(df_clean):,}")
    print(f"  • Columns: {len(df_clean.columns)}")
    print(f"  • NaN values: {df_clean.isna().sum().sum()}")
    
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
