"""
Macro Loader Module for Apple Financial and Social Analysis Dataset.
Fetches macroeconomic data from FRED and integrates with Apple stock analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 9)


def fetch_macro_data_from_fred():
    """
    Fetch macroeconomic data from FRED using pandas_datareader.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with macro data from FRED
    """
    print("Fetching macroeconomic data from FRED...")
    try:
        import pandas_datareader as pdr
        
        start_date = '2015-01-01'
        end_date = '2025-12-31'
        
        # Fetch Federal Funds Rate
        print("  Downloading Federal Funds Rate (FEDFUNDS)...")
        fed_rate = pdr.get_data_fred('FEDFUNDS', start_date, end_date)
        
        # Fetch Consumer Price Index
        print("  Downloading Consumer Price Index (CPIAUCSL)...")
        cpi = pdr.get_data_fred('CPIAUCSL', start_date, end_date)
        
        # Fetch Trade Weighted U.S. Dollar Index
        print("  Downloading USD Index (DTWEXBGS)...")
        usd_index = pdr.get_data_fred('DTWEXBGS', start_date, end_date)
        
        # Combine into single DataFrame
        macro_df = pd.DataFrame({
            'fed_rate': fed_rate['FEDFUNDS'],
            'us_cpi': cpi['CPIAUCSL'],
            'usd_index': usd_index['DTWEXBGS']
        })
        
        # Reset index to make date a column
        macro_df = macro_df.reset_index()
        macro_df.columns = ['date', 'fed_rate', 'us_cpi', 'usd_index']
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        
        print(f"✓ Downloaded macro data: {len(macro_df)} records")
        print(f"  Date range: {macro_df['date'].min().date()} to {macro_df['date'].max().date()}")
        
        return macro_df
    except ImportError:
        print("✗ pandas_datareader not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas_datareader', '-q'])
        return fetch_macro_data_from_fred()
    except Exception as e:
        print(f"✗ Error fetching macro data from FRED: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_to_daily_frequency(df):
    """
    Convert monthly macro data to daily frequency using forward-fill.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with macro data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily frequency
    """
    print("Converting to daily frequency...")
    try:
        # Set date as index
        df_indexed = df.set_index('date')
        
        # Create complete daily date range
        date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        df_reindexed = df_indexed.reindex(date_range)
        
        # Forward-fill missing values
        df_daily = df_reindexed.fillna(method='ffill')
        
        # Reset index
        df_daily = df_daily.reset_index()
        df_daily.columns = ['date', 'fed_rate', 'us_cpi', 'usd_index']
        
        print(f"✓ Converted to daily frequency: {len(df_daily)} records")
        
        return df_daily
    except Exception as e:
        print(f"✗ Error converting to daily frequency: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_apple_dataset(csv_path):
    """
    Load existing Apple analysis dataset.
    
    Parameters:
    -----------
    csv_path : str
        Path to Apple dataset CSV
    
    Returns:
    --------
    pd.DataFrame
        Apple dataset
    """
    print("Loading Apple analysis dataset...")
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✓ Loaded Apple dataset: {len(df)} records")
        
        return df
    except Exception as e:
        print(f"✗ Error loading Apple dataset: {e}")
        return None


def merge_datasets(apple_df, macro_df):
    """
    Merge Apple dataset with macroeconomic data on date.
    
    Parameters:
    -----------
    apple_df : pd.DataFrame
        Apple stock dataset
    macro_df : pd.DataFrame
        Macroeconomic dataset
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("Merging datasets on date...")
    try:
        # Ensure date columns are in same format
        apple_df['date'] = pd.to_datetime(apple_df['date']).dt.date
        macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date
        
        # Merge datasets
        merged_df = apple_df.merge(macro_df, on='date', how='inner')
        
        # Convert date back to datetime
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        
        print(f"✓ Merged dataset contains {len(merged_df)} records")
        print(f"  Columns: {list(merged_df.columns)}")
        
        return merged_df
    except Exception as e:
        print(f"✗ Error merging datasets: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_macro_sentiment_index(df):
    """
    Create normalized macro sentiment index.
    Formula: (usd_index / usd_index.max()) - (fed_rate / fed_rate.max()) - (us_cpi / us_cpi.max())
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset
    
    Returns:
    --------
    pd.DataFrame
        Dataset with new macro_sentiment_index column
    """
    print("Creating macro sentiment index...")
    try:
        # Normalize each component
        usd_normalized = df['usd_index'] / df['usd_index'].max()
        fed_normalized = df['fed_rate'] / df['fed_rate'].max()
        cpi_normalized = df['us_cpi'] / df['us_cpi'].max()
        
        # Create sentiment index
        df['macro_sentiment_index'] = usd_normalized - fed_normalized - cpi_normalized
        
        print(f"✓ Macro sentiment index created")
        print(f"  Min: {df['macro_sentiment_index'].min():.4f}")
        print(f"  Max: {df['macro_sentiment_index'].max():.4f}")
        print(f"  Mean: {df['macro_sentiment_index'].mean():.4f}")
        
        return df
    except Exception as e:
        print(f"✗ Error creating sentiment index: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_macro_dataset(df, output_path):
    """
    Save merged macro dataset to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Macro dataset
    output_path : str
        Path to save CSV
    """
    print("Saving macro dataset...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Macro dataset saved to {output_path}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")


def plot_macro_trend_comparison(df, output_dir):
    """
    Generate multi-axis line plot comparing macro indicators with Apple close price.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset
    output_dir : str
        Output directory for figure
    """
    print("Generating macro trend comparison plot...")
    try:
        fig, ax1 = plt.subplots(figsize=(16, 9))
        
        # Primary axis: Apple closing price
        color1 = '#1f77b4'
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Apple Stock Closing Price (USD)', fontsize=12, color=color1, fontweight='bold')
        line1 = ax1.plot(df['date'], df['close'], color=color1, linewidth=2.5, label='Apple Close Price')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis 1: Federal Funds Rate
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', 60))
        color2 = '#ff7f0e'
        ax2.set_ylabel('Federal Funds Rate (%)', fontsize=12, color=color2, fontweight='bold')
        line2 = ax2.plot(df['date'], df['fed_rate'], color=color2, linewidth=2, label='Fed Rate')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Secondary axis 2: Consumer Price Index
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 120))
        color3 = '#2ca02c'
        ax3.set_ylabel('Consumer Price Index', fontsize=12, color=color3, fontweight='bold')
        line3 = ax3.plot(df['date'], df['us_cpi'], color=color3, linewidth=2, label='US CPI')
        ax3.tick_params(axis='y', labelcolor=color3)
        
        # Secondary axis 3: USD Index
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 180))
        color4 = '#d62728'
        ax4.set_ylabel('Trade Weighted USD Index', fontsize=12, color=color4, fontweight='bold')
        line4 = ax4.plot(df['date'], df['usd_index'], color=color4, linewidth=2, label='USD Index')
        ax4.tick_params(axis='y', labelcolor=color4)
        
        # Title
        plt.title('Apple Stock Price vs. Macroeconomic Indicators (2015-2025)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'macro_trend_comparison.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Macro trend comparison plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating plot: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main execution function:
    1. Fetch macro data from FRED
    2. Convert to daily frequency
    3. Load Apple dataset
    4. Merge datasets
    5. Create sentiment index
    6. Save merged dataset
    7. Generate visualization
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL AND SOCIAL ANALYSIS - MACROECONOMIC DATA INTEGRATION")
    print("="*80 + "\n")
    
    # Define paths
    apple_csv = 'data/processed/apple_analysis_dataset.csv'
    output_csv = 'data/processed/apple_macro_dataset.csv'
    output_figures_dir = 'reports/figures'
    
    # Step 1: Fetch macro data
    print("[1/7] Fetching macroeconomic data...")
    macro_df = fetch_macro_data_from_fred()
    if macro_df is None:
        print("✗ Failed to fetch macro data. Exiting.")
        return False
    
    # Step 2: Convert to daily frequency
    print("\n[2/7] Converting to daily frequency...")
    macro_daily = convert_to_daily_frequency(macro_df)
    if macro_daily is None:
        print("✗ Failed to convert frequency. Exiting.")
        return False
    
    # Step 3: Load Apple dataset
    print("\n[3/7] Loading Apple dataset...")
    apple_df = load_apple_dataset(apple_csv)
    if apple_df is None:
        print("✗ Failed to load Apple dataset. Exiting.")
        return False
    
    # Step 4: Merge datasets
    print("\n[4/7] Merging datasets...")
    merged_df = merge_datasets(apple_df, macro_daily)
    if merged_df is None:
        print("✗ Failed to merge datasets. Exiting.")
        return False
    
    # Step 5: Create sentiment index
    print("\n[5/7] Creating macro sentiment index...")
    merged_df = create_macro_sentiment_index(merged_df)
    if merged_df is None:
        print("✗ Failed to create sentiment index. Exiting.")
        return False
    
    # Step 6: Save merged dataset
    print("\n[6/7] Saving merged dataset...")
    save_macro_dataset(merged_df, output_csv)
    
    # Step 7: Generate visualization
    print("\n[7/7] Generating visualization...")
    plot_macro_trend_comparison(merged_df, output_figures_dir)
    
    # Summary
    print("\n" + "="*80)
    print("MACRO DATA INTEGRATION COMPLETE")
    print("="*80)
    
    print(f"\n✓ Macro Data Sources:")
    print(f"  • FRED (Federal Reserve Economic Data)")
    print(f"  • Federal Funds Rate (FEDFUNDS)")
    print(f"  • Consumer Price Index (CPIAUCSL)")
    print(f"  • Trade Weighted USD Index (DTWEXBGS)")
    
    print(f"\n✓ Dataset Statistics:")
    print(f"  Records: {len(merged_df):,}")
    print(f"  Date Range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    print(f"  Columns: {len(merged_df.columns)}")
    
    print(f"\n✓ Macro Sentiment Index:")
    print(f"  Min: {merged_df['macro_sentiment_index'].min():.4f}")
    print(f"  Max: {merged_df['macro_sentiment_index'].max():.4f}")
    print(f"  Mean: {merged_df['macro_sentiment_index'].mean():.4f}")
    print(f"  Std: {merged_df['macro_sentiment_index'].std():.4f}")
    
    print(f"\n✓ Output Files:")
    print(f"  • {output_csv}")
    print(f"  • {output_figures_dir}/macro_trend_comparison.png")
    
    print(f"\n✓ New Columns:")
    print(f"  • fed_rate: Federal Funds Rate (%)")
    print(f"  • us_cpi: Consumer Price Index")
    print(f"  • usd_index: Trade Weighted USD Index")
    print(f"  • macro_sentiment_index: Normalized combination")
    
    print("\n" + "="*80)
    print("✓ All macro data integration tasks completed successfully!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
