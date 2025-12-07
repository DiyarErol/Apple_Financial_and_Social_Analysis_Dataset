"""
Macro Extension Script for Apple Financial and Social Analysis Dataset.
Fetches additional macroeconomic indicators (VIX, Gold, Oil, Treasury, Unemployment).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yfinance as yf
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def download_yfinance_data(tickers, start_date, end_date):
    """
    Download daily data from yfinance.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns:
    --------
    dict
        Dictionary with ticker as key and DataFrame as value
    """
    print(f"Downloading yfinance data for {len(tickers)} tickers...")
    print(f"  Tickers: {', '.join(tickers)}")
    
    data_dict = {}
    
    for ticker in tickers:
        try:
            print(f"  Fetching {ticker}...", end=' ')
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(df) > 0:
                # Extract close prices - handle MultiIndex and single ticker
                # For single ticker, Close is a Series
                if 'Close' in df.columns:
                    close_data = df['Close']
                    # If MultiIndex, flatten it
                    if hasattr(close_data, 'columns'):
                        close_data = close_data.iloc[:, 0]
                    
                    data_dict[ticker] = pd.DataFrame({
                        'date': df.index,
                        'value': close_data.values
                    })
                    print(f"✓ {len(data_dict[ticker])} records")
                else:
                    print(f"✗ No Close column")
            else:
                print(f"✗ No data")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    return data_dict


def download_fred_data(series_codes, start_date, end_date):
    """
    Download data from FRED.
    
    Parameters:
    -----------
    series_codes : list
        List of FRED series codes
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns:
    --------
    dict
        Dictionary with series code as key and DataFrame as value
    """
    print(f"\nDownloading FRED data for {len(series_codes)} series...")
    print(f"  Series: {', '.join(series_codes)}")
    
    data_dict = {}
    
    try:
        import pandas_datareader as pdr
        
        for code in series_codes:
            try:
                print(f"  Fetching {code}...", end=' ')
                df = pdr.DataReader(code, 'fred', start_date, end_date)
                
                if len(df) > 0:
                    data_dict[code] = pd.DataFrame({
                        'date': df.index,
                        'value': df[code].values
                    })
                    print(f"✓ {len(data_dict[code])} records")
                else:
                    print(f"✗ No data")
            
            except Exception as e:
                print(f"✗ Error: {e}")
    
    except ImportError:
        print("✗ pandas_datareader not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas-datareader'])
        print("✓ pandas_datareader installed. Please run the script again.")
        return None
    
    return data_dict


def resample_to_daily(df, method='ffill'):
    """
    Resample monthly/weekly data to daily frequency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date and value columns
    method : str
        Resampling method ('ffill' or 'interpolate')
    
    Returns:
    --------
    pd.DataFrame
        Daily frequency DataFrame
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Create daily date range
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    
    # Reindex and forward fill
    df_daily = df.reindex(date_range)
    
    if method == 'ffill':
        df_daily = df_daily.ffill()
    elif method == 'interpolate':
        df_daily = df_daily.interpolate()
    
    df_daily = df_daily.reset_index()
    df_daily.columns = ['date', 'value']
    
    return df_daily


def fetch_all_macro_data(start_date, end_date):
    """
    Fetch all macroeconomic data.
    
    Parameters:
    -----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame
        Combined macro data
    """
    print("\n" + "="*80)
    print("FETCHING MACROECONOMIC DATA")
    print("="*80 + "\n")
    
    # yfinance tickers
    yf_tickers = ['^VIX', 'GC=F', 'CL=F', '^TNX']
    yf_data = download_yfinance_data(yf_tickers, start_date, end_date)
    
    # FRED series
    fred_series = ['DGS10', 'UNRATE']
    fred_data = download_fred_data(fred_series, start_date, end_date)
    
    if fred_data is None:
        print("\n✗ Failed to download FRED data. Exiting.")
        return None
    
    # Process data
    print("\nProcessing macroeconomic data...")
    
    macro_df = None
    
    # VIX
    if '^VIX' in yf_data:
        vix_df = yf_data['^VIX'].copy()
        vix_df.columns = ['date', 'vix']
        macro_df = vix_df
        print(f"  ✓ VIX: {len(vix_df)} records")
    
    # Gold
    if 'GC=F' in yf_data:
        gold_df = yf_data['GC=F'].copy()
        gold_df.columns = ['date', 'gold']
        if macro_df is not None:
            macro_df = macro_df.merge(gold_df, on='date', how='outer')
        else:
            macro_df = gold_df
        print(f"  ✓ Gold: {len(gold_df)} records")
    
    # Oil (WTI)
    if 'CL=F' in yf_data:
        oil_df = yf_data['CL=F'].copy()
        oil_df.columns = ['date', 'oil']
        if macro_df is not None:
            macro_df = macro_df.merge(oil_df, on='date', how='outer')
        else:
            macro_df = oil_df
        print(f"  ✓ Oil (WTI): {len(oil_df)} records")
    
    # 10Y Treasury (prefer FRED DGS10)
    if 'DGS10' in fred_data and len(fred_data['DGS10']) > 0:
        tnote_df = fred_data['DGS10'].copy()
        tnote_df = resample_to_daily(tnote_df, method='ffill')
        tnote_df.columns = ['date', 'tnote_10y']
        if macro_df is not None:
            macro_df = macro_df.merge(tnote_df, on='date', how='outer')
        else:
            macro_df = tnote_df
        print(f"  ✓ 10Y Treasury (FRED): {len(tnote_df)} records (resampled to daily)")
    elif '^TNX' in yf_data:
        # Fallback to ^TNX from yfinance
        tnote_df = yf_data['^TNX'].copy()
        tnote_df['value'] = tnote_df['value'] * 0.1  # Convert to decimal
        tnote_df.columns = ['date', 'tnote_10y']
        if macro_df is not None:
            macro_df = macro_df.merge(tnote_df, on='date', how='outer')
        else:
            macro_df = tnote_df
        print(f"  ✓ 10Y Treasury (^TNX fallback): {len(tnote_df)} records")
    
    # Unemployment
    if 'UNRATE' in fred_data and len(fred_data['UNRATE']) > 0:
        unrate_df = fred_data['UNRATE'].copy()
        unrate_df = resample_to_daily(unrate_df, method='ffill')
        unrate_df.columns = ['date', 'unemployment']
        if macro_df is not None:
            macro_df = macro_df.merge(unrate_df, on='date', how='outer')
        else:
            macro_df = unrate_df
        print(f"  ✓ Unemployment: {len(unrate_df)} records (resampled to daily)")
    
    # Sort by date
    if macro_df is not None:
        macro_df = macro_df.sort_values('date').reset_index(drop=True)
        print(f"\n✓ Combined macro data: {len(macro_df)} records, {len(macro_df.columns)} columns")
        print(f"  Date range: {macro_df['date'].min().date()} to {macro_df['date'].max().date()}")
        print(f"  Columns: {list(macro_df.columns)}")
    
    return macro_df


def load_apple_dataset(csv_path):
    """
    Load Apple dataset with technical indicators and competitors.
    
    Parameters:
    -----------
    csv_path : str
        Path to the dataset CSV
    
    Returns:
    --------
    pd.DataFrame
        Apple dataset
    """
    print("\nLoading Apple dataset...")
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✓ Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_datasets(apple_df, macro_df):
    """
    Merge Apple dataset with macro data on date.
    
    Parameters:
    -----------
    apple_df : pd.DataFrame
        Apple dataset
    macro_df : pd.DataFrame
        Macro data
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("\nMerging datasets...")
    try:
        # Left join on date
        merged_df = apple_df.merge(macro_df, on='date', how='left')
        
        print(f"✓ Datasets merged")
        print(f"  Total rows: {len(merged_df)}")
        print(f"  Total columns: {len(merged_df.columns)}")
        print(f"  NaN values: {merged_df.isna().sum().sum()}")
        
        # Fill NaN with forward fill for macro data
        macro_cols = ['vix', 'gold', 'oil', 'tnote_10y', 'unemployment']
        for col in macro_cols:
            if col in merged_df.columns:
                nan_count = merged_df[col].isna().sum()
                if nan_count > 0:
                    merged_df[col] = merged_df[col].ffill()
                    print(f"  Forward-filled {nan_count} NaN values in {col}")
        
        return merged_df
    
    except Exception as e:
        print(f"✗ Error merging datasets: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    print("\nSaving dataset...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset saved to {output_path}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")


def normalize_column(series):
    """
    Normalize a series to 0-1 range.
    
    Parameters:
    -----------
    series : pd.Series
        Series to normalize
    
    Returns:
    --------
    pd.Series
        Normalized series
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val - min_val == 0:
        return series * 0
    
    return (series - min_val) / (max_val - min_val)


def plot_macro_overlay(df, output_dir):
    """
    Generate macro overlay plot with scaled indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with macro data
    output_dir : str
        Output directory for the plot
    """
    print("\nGenerating macro overlay plot...")
    try:
        # Check which columns are available
        available_cols = ['date', 'close']
        optional_cols = ['vix', 'gold', 'oil', 'tnote_10y']
        
        for col in optional_cols:
            if col in df.columns:
                available_cols.append(col)
        
        if len(available_cols) < 3:
            print(f"✗ Insufficient data for plotting. Available: {available_cols}")
            return
        
        # Clean data for plotting
        plot_df = df[available_cols].dropna().copy()
        
        if len(plot_df) == 0:
            print("✗ No data available for plotting")
            return
        
        # Normalize all available indicators to 0-1
        plot_df['norm_close'] = normalize_column(plot_df['close'])
        
        if 'vix' in plot_df.columns:
            plot_df['norm_vix'] = normalize_column(plot_df['vix'])
        if 'gold' in plot_df.columns:
            plot_df['norm_gold'] = normalize_column(plot_df['gold'])
        if 'oil' in plot_df.columns:
            plot_df['norm_oil'] = normalize_column(plot_df['oil'])
        if 'tnote_10y' in plot_df.columns:
            plot_df['norm_tnote'] = normalize_column(plot_df['tnote_10y'])
        
        # Create plot with two panels
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Panel 1: Close price with available indicators (VIX, Gold, Treasury)
        ax1 = axes[0]
        ax1.plot(plot_df['date'], plot_df['norm_close'], color='#1f77b4', 
                linewidth=2.5, label='AAPL Close (normalized)', zorder=3)
        
        if 'norm_vix' in plot_df.columns:
            ax1.plot(plot_df['date'], plot_df['norm_vix'], color='#ff7f0e', 
                    linewidth=2, label='VIX (normalized)', alpha=0.7)
        if 'norm_gold' in plot_df.columns:
            ax1.plot(plot_df['date'], plot_df['norm_gold'], color='#d4af37', 
                    linewidth=2, label='Gold (normalized)', alpha=0.7)
        if 'norm_tnote' in plot_df.columns and 'norm_vix' not in plot_df.columns:
            ax1.plot(plot_df['date'], plot_df['norm_tnote'], color='#d62728', 
                    linewidth=2, label='10Y Treasury (normalized)', alpha=0.7)
        
        ax1.set_ylabel('Normalized Value (0-1)', fontsize=11, fontweight='bold')
        ax1.set_title('Apple Stock Price vs Macroeconomic Indicators (Normalized 0-1)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.05, 1.05])
        
        # Panel 2: Close price with Oil and Treasury
        ax2 = axes[1]
        ax2.plot(plot_df['date'], plot_df['norm_close'], color='#1f77b4', 
                linewidth=2.5, label='AAPL Close (normalized)', zorder=3)
        
        if 'norm_oil' in plot_df.columns:
            ax2.plot(plot_df['date'], plot_df['norm_oil'], color='#2ca02c', 
                    linewidth=2, label='WTI Crude Oil (normalized)', alpha=0.7)
        if 'norm_tnote' in plot_df.columns:
            ax2.plot(plot_df['date'], plot_df['norm_tnote'], color='#d62728', 
                    linewidth=2, label='10Y Treasury (normalized)', alpha=0.7)
        
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Normalized Value (0-1)', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'macro_overlay.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Macro overlay plot saved to {output_path}")
        plt.close()
    
    except Exception as e:
        print(f"✗ Error generating macro overlay plot: {e}")
        import traceback
        traceback.print_exc()


def print_macro_statistics(df):
    """
    Print statistics for macro indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with macro data
    """
    print("\n" + "="*80)
    print("MACRO INDICATOR STATISTICS")
    print("="*80 + "\n")
    
    macro_cols = ['vix', 'gold', 'oil', 'tnote_10y', 'unemployment']
    
    for col in macro_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                print(f"{col}:")
                print(f"  Count:    {len(valid_data)}")
                print(f"  Mean:     {valid_data.mean():.4f}")
                print(f"  Median:   {valid_data.median():.4f}")
                print(f"  Std:      {valid_data.std():.4f}")
                print(f"  Min:      {valid_data.min():.4f}")
                print(f"  Max:      {valid_data.max():.4f}")
                print()


def main():
    """
    Main execution function:
    1. Fetch macro data (VIX, Gold, Oil, Treasury, Unemployment)
    2. Resample FRED monthly data to daily
    3. Load Apple dataset
    4. Merge datasets
    5. Save to CSV
    6. Generate visualization
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL ANALYSIS - MACRO DATA EXTENSION")
    print("="*80 + "\n")
    
    # Define parameters
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    
    apple_csv = 'data/processed/apple_with_ti_cmp.csv'
    output_csv = 'data/processed/apple_feature_plus.csv'
    figures_dir = 'reports/figures'
    
    # Step 1: Fetch macro data
    print("[1/5] Fetching macroeconomic data...")
    macro_df = fetch_all_macro_data(start_date, end_date)
    if macro_df is None:
        print("✗ Failed to fetch macro data. Exiting.")
        return False
    
    # Step 2: Load Apple dataset
    print("\n[2/5] Loading Apple dataset...")
    apple_df = load_apple_dataset(apple_csv)
    if apple_df is None:
        print("✗ Failed to load Apple dataset. Exiting.")
        return False
    
    # Step 3: Merge datasets
    print("\n[3/5] Merging datasets...")
    merged_df = merge_datasets(apple_df, macro_df)
    if merged_df is None:
        print("✗ Failed to merge datasets. Exiting.")
        return False
    
    # Step 4: Save dataset
    print("\n[4/5] Saving extended dataset...")
    save_dataset(merged_df, output_csv)
    
    # Step 5: Generate visualization
    print("\n[5/5] Generating visualization...")
    plot_macro_overlay(merged_df, figures_dir)
    
    # Print statistics
    print_macro_statistics(merged_df)
    
    # Final summary
    print("="*80)
    print("MACRO DATA EXTENSION COMPLETE")
    print("="*80)
    
    print(f"\n✓ Output Files:")
    print(f"  • {output_csv}")
    print(f"  • {figures_dir}/macro_overlay.png")
    
    print(f"\n✓ Macro Indicators Added:")
    print(f"  • VIX (Volatility Index)")
    print(f"  • Gold Futures (GC=F)")
    print(f"  • WTI Crude Oil (CL=F)")
    print(f"  • 10Y Treasury Yield")
    print(f"  • Unemployment Rate")
    
    print(f"\n✓ Data Quality:")
    print(f"  • Records: {len(merged_df):,}")
    print(f"  • Columns: {len(merged_df.columns)}")
    print(f"  • Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
