"""
Lag Features Script for Apple Financial and Social Analysis Dataset.
Adds lagged and rolling window features for time series modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pandas.plotting import lag_plot, autocorrelation_plot
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def load_dataset(csv_path):
    """
    Load Apple dataset with temporal features.
    
    Parameters:
    -----------
    csv_path : str
        Path to the dataset CSV
    
    Returns:
    --------
    pd.DataFrame
        Dataset with datetime converted
    """
    print("Loading dataset...")
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_lag_features(df):
    """
    Add lagged features for time series modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset sorted by date
    
    Returns:
    --------
    pd.DataFrame
        Dataset with lag features added
    """
    print("\nAdding lag features...")
    try:
        df_lag = df.copy()
        
        # Close price lags
        df_lag['close_lag_1'] = df_lag['close'].shift(1)
        print(f"  ✓ close_lag_1: 1-day lagged close price")
        
        df_lag['close_lag_2'] = df_lag['close'].shift(2)
        print(f"  ✓ close_lag_2: 2-day lagged close price")
        
        df_lag['close_lag_7'] = df_lag['close'].shift(7)
        print(f"  ✓ close_lag_7: 7-day lagged close price")
        
        # Volume lag
        df_lag['volume_lag_1'] = df_lag['volume'].shift(1)
        print(f"  ✓ volume_lag_1: 1-day lagged volume")
        
        # Return lags
        if 'daily_return' in df_lag.columns:
            df_lag['return_lag_1'] = df_lag['daily_return'].shift(1)
            print(f"  ✓ return_lag_1: 1-day lagged daily return")
            
            df_lag['return_lag_7'] = df_lag['daily_return'].shift(7)
            print(f"  ✓ return_lag_7: 7-day lagged daily return")
        else:
            print(f"  ⚠ daily_return column not found, skipping return lags")
        
        # Rolling max and min (30-day)
        df_lag['rolling_max_30d'] = df_lag['close'].rolling(window=30).max()
        print(f"  ✓ rolling_max_30d: 30-day rolling maximum")
        
        df_lag['rolling_min_30d'] = df_lag['close'].rolling(window=30).min()
        print(f"  ✓ rolling_min_30d: 30-day rolling minimum")
        
        print(f"\n✓ Lag features added: 8 new columns")
        
        return df_lag
    
    except Exception as e:
        print(f"✗ Error adding lag features: {e}")
        import traceback
        traceback.print_exc()
        return df


def clean_data(df):
    """
    Remove rows with NaN values introduced by shifting/rolling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with lag features
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("\nCleaning data...")
    try:
        initial_rows = len(df)
        
        # Only drop rows where lag features have NaN (not other columns)
        lag_cols = ['close_lag_1', 'close_lag_2', 'close_lag_7', 'volume_lag_1',
                   'return_lag_1', 'return_lag_7', 'rolling_max_30d', 'rolling_min_30d']
        
        # Drop rows where any lag column has NaN
        df_clean = df.dropna(subset=lag_cols)
        
        rows_dropped = initial_rows - len(df_clean)
        
        print(f"✓ Data cleaned")
        print(f"  Initial rows: {initial_rows}")
        print(f"  Rows dropped: {rows_dropped}")
        print(f"  Final rows: {len(df_clean)}")
        
        if len(df_clean) > 0:
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
    print("\nSaving dataset...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset saved to {output_path}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")


def plot_lag_diagnostics(df, output_dir):
    """
    Generate lag diagnostics plots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with lag features
    output_dir : str
        Output directory for the plot
    """
    print("\nGenerating lag diagnostics plot...")
    try:
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Layout: 3 rows x 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Row 1: Lag plots for close price
        ax1 = fig.add_subplot(gs[0, 0])
        lag_plot(df['close'], lag=1, ax=ax1, c='#1f77b4', alpha=0.5, s=10)
        ax1.set_title('Lag-1 Plot: Close Price', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Close(t)', fontsize=10)
        ax1.set_ylabel('Close(t+1)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        lag_plot(df['close'], lag=2, ax=ax2, c='#ff7f0e', alpha=0.5, s=10)
        ax2.set_title('Lag-2 Plot: Close Price', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Close(t)', fontsize=10)
        ax2.set_ylabel('Close(t+2)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        lag_plot(df['close'], lag=7, ax=ax3, c='#2ca02c', alpha=0.5, s=10)
        ax3.set_title('Lag-7 Plot: Close Price', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Close(t)', fontsize=10)
        ax3.set_ylabel('Close(t+7)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Row 2: Lag plots for daily return
        if 'daily_return' in df.columns:
            ax4 = fig.add_subplot(gs[1, 0])
            daily_return_clean = df['daily_return'].dropna()
            lag_plot(daily_return_clean, lag=1, ax=ax4, c='#d62728', alpha=0.5, s=10)
            ax4.set_title('Lag-1 Plot: Daily Return', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Return(t)', fontsize=10)
            ax4.set_ylabel('Return(t+1)', fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            ax5 = fig.add_subplot(gs[1, 1])
            lag_plot(daily_return_clean, lag=5, ax=ax5, c='#9467bd', alpha=0.5, s=10)
            ax5.set_title('Lag-5 Plot: Daily Return', fontsize=11, fontweight='bold')
            ax5.set_xlabel('Return(t)', fontsize=10)
            ax5.set_ylabel('Return(t+5)', fontsize=10)
            ax5.grid(True, alpha=0.3)
            
            ax6 = fig.add_subplot(gs[1, 2])
            lag_plot(daily_return_clean, lag=10, ax=ax6, c='#8c564b', alpha=0.5, s=10)
            ax6.set_title('Lag-10 Plot: Daily Return', fontsize=11, fontweight='bold')
            ax6.set_xlabel('Return(t)', fontsize=10)
            ax6.set_ylabel('Return(t+10)', fontsize=10)
            ax6.grid(True, alpha=0.3)
        
        # Row 3: Autocorrelation plots
        ax7 = fig.add_subplot(gs[2, :2])
        autocorrelation_plot(df['close'], ax=ax7, color='#1f77b4', linewidth=2)
        ax7.set_title('Autocorrelation Plot: Close Price', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Lag', fontsize=10)
        ax7.set_ylabel('Autocorrelation', fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        # Rolling statistics plot
        ax8 = fig.add_subplot(gs[2, 2])
        
        # Calculate rolling mean and std for visualization
        rolling_mean = df['close'].rolling(window=30).mean()
        rolling_std = df['close'].rolling(window=30).std()
        
        ax8.plot(df.index[-200:], df['close'].iloc[-200:], 
                label='Close', color='#1f77b4', linewidth=1.5, alpha=0.7)
        ax8.plot(df.index[-200:], rolling_mean.iloc[-200:], 
                label='30d Mean', color='#ff7f0e', linewidth=2)
        ax8.fill_between(df.index[-200:], 
                        (rolling_mean - rolling_std).iloc[-200:],
                        (rolling_mean + rolling_std).iloc[-200:],
                        alpha=0.2, color='#ff7f0e')
        ax8.set_title('Rolling Mean ± Std (Last 200 days)', fontsize=11, fontweight='bold')
        ax8.set_xlabel('Index', fontsize=10)
        ax8.set_ylabel('Price (USD)', fontsize=10)
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)
        
        fig.suptitle('Lag Diagnostics: Apple Stock Price Time Series Analysis', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'lag_diagnostics.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Lag diagnostics plot saved to {output_path}")
        plt.close()
    
    except Exception as e:
        print(f"✗ Error generating lag diagnostics plot: {e}")
        import traceback
        traceback.print_exc()


def print_lag_statistics(df):
    """
    Print statistics for lag features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with lag features
    """
    print("\n" + "="*80)
    print("LAG FEATURE STATISTICS")
    print("="*80 + "\n")
    
    lag_cols = ['close_lag_1', 'close_lag_2', 'close_lag_7', 'volume_lag_1',
                'return_lag_1', 'return_lag_7', 'rolling_max_30d', 'rolling_min_30d']
    
    for col in lag_cols:
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
    
    # Correlation analysis
    print("\nCorrelation with Current Close Price:")
    if 'close' in df.columns:
        for col in ['close_lag_1', 'close_lag_2', 'close_lag_7']:
            if col in df.columns:
                corr = df['close'].corr(df[col])
                print(f"  {col}: {corr:.4f}")
    
    if 'daily_return' in df.columns:
        print("\nCorrelation of Returns (Serial Correlation):")
        if 'return_lag_1' in df.columns:
            corr1 = df['daily_return'].corr(df['return_lag_1'])
            print(f"  return_lag_1: {corr1:.4f}")
        if 'return_lag_7' in df.columns:
            corr7 = df['daily_return'].corr(df['return_lag_7'])
            print(f"  return_lag_7: {corr7:.4f}")


def main():
    """
    Main execution function:
    1. Load dataset
    2. Add lag features
    3. Clean data (drop NaN)
    4. Save dataset
    5. Generate visualizations
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL ANALYSIS - LAG FEATURES ENGINEERING")
    print("="*80 + "\n")
    
    # Define paths
    input_csv = 'data/processed/apple_feature_plus_time.csv'
    output_csv = 'data/processed/apple_feature_plus_time_lag.csv'
    figures_dir = 'reports/figures'
    
    # Step 1: Load dataset
    print("[1/5] Loading dataset...")
    df = load_dataset(input_csv)
    if df is None:
        print("✗ Failed to load dataset. Exiting.")
        return False
    
    # Step 2: Add lag features
    print("\n[2/5] Adding lag features...")
    df_lag = add_lag_features(df)
    
    # Step 3: Clean data
    print("\n[3/5] Cleaning data...")
    df_clean = clean_data(df_lag)
    
    # Step 4: Save dataset
    print("\n[4/5] Saving dataset...")
    save_dataset(df_clean, output_csv)
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    plot_lag_diagnostics(df_clean, figures_dir)
    
    # Print statistics
    print_lag_statistics(df_clean)
    
    # Final summary
    print("\n" + "="*80)
    print("LAG FEATURES ENGINEERING COMPLETE")
    print("="*80)
    
    print(f"\n✓ Output Files:")
    print(f"  • {output_csv}")
    print(f"  • {figures_dir}/lag_diagnostics.png")
    
    print(f"\n✓ Lag Features Added:")
    print(f"  • close_lag_1, close_lag_2, close_lag_7: Lagged close prices")
    print(f"  • volume_lag_1: Lagged volume")
    print(f"  • return_lag_1, return_lag_7: Lagged daily returns")
    print(f"  • rolling_max_30d, rolling_min_30d: 30-day rolling extrema")
    
    print(f"\n✓ Data Quality:")
    print(f"  • Records: {len(df_clean):,}")
    print(f"  • Columns: {len(df_clean.columns)}")
    print(f"  • Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    print(f"  • NaN values: {df_clean.isna().sum().sum()}")
    
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
