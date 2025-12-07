"""
Competitors Loader Script for Apple Financial and Social Analysis Dataset.
Downloads competitor stock data and calculates relative performance metrics.
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


def download_competitor_data(tickers, start_date, end_date):
    """
    Download daily close prices for competitor stocks.
    
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
    pd.DataFrame
        DataFrame with close prices for all tickers
    """
    print(f"Downloading competitor data for {len(tickers)} tickers...")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Date range: {start_date} to {end_date}")
    
    try:
        # Download data for all tickers
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Extract close prices
        if len(tickers) == 1:
            close_prices = pd.DataFrame({'date': data.index, tickers[0]: data['Close'].values})
        else:
            close_prices = data['Close'].copy()
            close_prices = close_prices.reset_index()
        
        # Rename columns to clean format
        # First column is 'Date' or 'date', rest are tickers
        new_columns = ['date']
        for col in close_prices.columns:
            if col not in ['Date', 'date']:
                # Clean ticker names (remove ^)
                clean_name = col.replace('^', '')
                new_columns.append(f'close_{clean_name}')
        
        close_prices.columns = new_columns
        
        print(f"✓ Downloaded {len(close_prices)} daily records")
        print(f"  Columns: {list(close_prices.columns)}")
        print(f"  Date range: {close_prices['date'].min().date()} to {close_prices['date'].max().date()}")
        
        return close_prices
    
    except Exception as e:
        print(f"✗ Error downloading competitor data: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_relative_performance(df, base_col='close', comparison_cols=None):
    """
    Calculate relative performance metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    base_col : str
        Base column name (AAPL close price)
    comparison_cols : list
        List of column names to compare against
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with relative performance columns added
    """
    print("Calculating relative performance metrics...")
    
    try:
        df_rel = df.copy()
        
        if comparison_cols is None:
            comparison_cols = ['close_MSFT', 'close_IXIC']
        
        for comp_col in comparison_cols:
            if comp_col in df_rel.columns and base_col in df_rel.columns:
                # Create relative performance column
                rel_col_name = f"rel_AAPL_{comp_col.replace('close_', '')}"
                df_rel[rel_col_name] = df_rel[base_col] / df_rel[comp_col]
                
                print(f"  ✓ {rel_col_name}: range {df_rel[rel_col_name].min():.4f} to {df_rel[rel_col_name].max():.4f}")
        
        print(f"✓ Relative performance metrics calculated")
        
        return df_rel
    
    except Exception as e:
        print(f"✗ Error calculating relative performance: {e}")
        import traceback
        traceback.print_exc()
        return df


def load_apple_ti_dataset(csv_path):
    """
    Load Apple dataset with technical indicators.
    
    Parameters:
    -----------
    csv_path : str
        Path to the technical indicators dataset CSV
    
    Returns:
    --------
    pd.DataFrame
        Technical indicators dataset
    """
    print("Loading Apple dataset with technical indicators...")
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


def merge_datasets(apple_df, competitor_df):
    """
    Merge Apple dataset with competitor data on date.
    
    Parameters:
    -----------
    apple_df : pd.DataFrame
        Apple dataset with technical indicators
    competitor_df : pd.DataFrame
        Competitor price data
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("Merging datasets...")
    try:
        # Left join on date
        merged_df = apple_df.merge(competitor_df, on='date', how='left')
        
        print(f"✓ Datasets merged")
        print(f"  Total rows: {len(merged_df)}")
        print(f"  Total columns: {len(merged_df.columns)}")
        print(f"  NaN values: {merged_df.isna().sum().sum()}")
        
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
    print("Saving dataset...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset saved to {output_path}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")


def plot_relative_performance(df, output_dir):
    """
    Generate relative performance plot (normalized to 1.0 at start).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with competitor data
    output_dir : str
        Output directory for the plot
    """
    print("Generating relative performance plot...")
    try:
        # Create a clean dataframe for plotting
        plot_df = df[['date', 'close', 'close_MSFT', 'close_XLK']].dropna().copy()
        
        if len(plot_df) == 0:
            print("✗ No data available for plotting")
            return
        
        # Normalize to 1.0 at start
        first_aapl = plot_df['close'].iloc[0]
        first_msft = plot_df['close_MSFT'].iloc[0]
        first_xlk = plot_df['close_XLK'].iloc[0]
        
        plot_df['norm_AAPL'] = plot_df['close'] / first_aapl
        plot_df['norm_MSFT'] = plot_df['close_MSFT'] / first_msft
        plot_df['norm_XLK'] = plot_df['close_XLK'] / first_xlk
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 9))
        
        ax.plot(plot_df['date'], plot_df['norm_AAPL'], color='#1f77b4', 
               linewidth=2.5, label='AAPL (Apple)', zorder=3)
        ax.plot(plot_df['date'], plot_df['norm_MSFT'], color='#ff7f0e', 
               linewidth=2, label='MSFT (Microsoft)', alpha=0.85)
        ax.plot(plot_df['date'], plot_df['norm_XLK'], color='#2ca02c', 
               linewidth=2, label='XLK (Technology ETF)', alpha=0.85)
        
        # Add horizontal line at 1.0
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='Starting Point')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Price (Starting Point = 1.0)', fontsize=12, fontweight='bold')
        ax.set_title('Apple vs Competitors: Relative Performance (Normalized)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        # Add performance statistics box
        final_aapl = plot_df['norm_AAPL'].iloc[-1]
        final_msft = plot_df['norm_MSFT'].iloc[-1]
        final_xlk = plot_df['norm_XLK'].iloc[-1]
        
        stats_text = f'Final Performance (vs Start):\n'
        stats_text += f'AAPL: {(final_aapl - 1) * 100:+.1f}%\n'
        stats_text += f'MSFT: {(final_msft - 1) * 100:+.1f}%\n'
        stats_text += f'XLK:  {(final_xlk - 1) * 100:+.1f}%'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'competitors_relative_perf.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Relative performance plot saved to {output_path}")
        plt.close()
    
    except Exception as e:
        print(f"✗ Error generating relative performance plot: {e}")
        import traceback
        traceback.print_exc()


def print_competitor_statistics(df):
    """
    Print statistics for competitor data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with competitor data
    """
    print("\n" + "="*80)
    print("COMPETITOR DATA STATISTICS")
    print("="*80 + "\n")
    
    competitor_cols = [col for col in df.columns if col.startswith('close_') or col.startswith('rel_')]
    
    for col in competitor_cols:
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
    1. Download competitor data (AAPL, MSFT, GOOGL, AMZN, ^IXIC, XLK)
    2. Calculate relative performance metrics
    3. Load Apple dataset with technical indicators
    4. Merge datasets
    5. Save to CSV
    6. Generate visualization
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL ANALYSIS - COMPETITOR DATA LOADING")
    print("="*80 + "\n")
    
    # Define parameters
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', '^IXIC', 'XLK']
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    
    apple_ti_csv = 'data/processed/apple_with_ti.csv'
    output_csv = 'data/processed/apple_with_ti_cmp.csv'
    figures_dir = 'reports/figures'
    
    # Step 1: Download competitor data
    print("[1/6] Downloading competitor data...")
    competitor_df = download_competitor_data(tickers, start_date, end_date)
    if competitor_df is None:
        print("✗ Failed to download competitor data. Exiting.")
        return False
    
    # Step 2: Calculate relative performance
    print("\n[2/6] Calculating relative performance metrics...")
    competitor_df = calculate_relative_performance(
        competitor_df, 
        base_col='close_AAPL', 
        comparison_cols=['close_MSFT', 'close_IXIC']
    )
    
    # Step 3: Load Apple dataset with technical indicators
    print("\n[3/6] Loading Apple dataset with technical indicators...")
    apple_df = load_apple_ti_dataset(apple_ti_csv)
    if apple_df is None:
        print("✗ Failed to load Apple dataset. Exiting.")
        return False
    
    # Step 4: Merge datasets
    print("\n[4/6] Merging datasets...")
    merged_df = merge_datasets(apple_df, competitor_df)
    if merged_df is None:
        print("✗ Failed to merge datasets. Exiting.")
        return False
    
    # Step 5: Save dataset
    print("\n[5/6] Saving merged dataset...")
    save_dataset(merged_df, output_csv)
    
    # Step 6: Generate visualization
    print("\n[6/6] Generating visualization...")
    plot_relative_performance(merged_df, figures_dir)
    
    # Print statistics
    print_competitor_statistics(merged_df)
    
    # Final summary
    print("="*80)
    print("COMPETITOR DATA LOADING COMPLETE")
    print("="*80)
    
    print(f"\n✓ Output Files:")
    print(f"  • {output_csv}")
    print(f"  • {figures_dir}/competitors_relative_perf.png")
    
    print(f"\n✓ Competitor Data Added:")
    print(f"  • MSFT (Microsoft)")
    print(f"  • GOOGL (Google)")
    print(f"  • AMZN (Amazon)")
    print(f"  • ^IXIC (Nasdaq Composite)")
    print(f"  • XLK (Technology ETF)")
    
    print(f"\n✓ Relative Performance Metrics:")
    print(f"  • rel_AAPL_MSFT (AAPL / MSFT)")
    print(f"  • rel_AAPL_IXIC (AAPL / Nasdaq)")
    
    print(f"\n✓ Data Quality:")
    print(f"  • Records: {len(merged_df):,}")
    print(f"  • Columns: {len(merged_df.columns)}")
    print(f"  • Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
