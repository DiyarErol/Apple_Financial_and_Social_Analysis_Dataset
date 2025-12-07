"""
Temporal Features Script for Apple Financial and Social Analysis Dataset.
Adds time-based features for seasonality and calendar effects analysis.
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


def load_dataset(csv_path):
    """
    Load Apple dataset with extended features.
    
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
        
        print(f"✓ Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_temporal_features(df):
    """
    Add temporal/time-based features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with date column
    
    Returns:
    --------
    pd.DataFrame
        Dataset with temporal features added
    """
    print("\nAdding temporal features...")
    try:
        df_time = df.copy()
        
        # Day of week (0=Monday, 6=Sunday)
        df_time['dow'] = df_time['date'].dt.dayofweek
        print(f"  ✓ dow (day of week): 0-6 (0=Monday, 6=Sunday)")
        
        # Month (1-12)
        df_time['month'] = df_time['date'].dt.month
        print(f"  ✓ month: 1-12")
        
        # Quarter (1-4)
        df_time['quarter'] = df_time['date'].dt.quarter
        print(f"  ✓ quarter: 1-4")
        
        # Year
        df_time['year'] = df_time['date'].dt.year
        print(f"  ✓ year: {df_time['year'].min()} to {df_time['year'].max()}")
        
        # Is month end (boolean)
        df_time['is_month_end'] = df_time['date'].dt.is_month_end.astype(int)
        month_end_count = df_time['is_month_end'].sum()
        print(f"  ✓ is_month_end: {month_end_count} month-end dates")
        
        # Is quarter end (boolean)
        df_time['is_quarter_end'] = df_time['date'].dt.is_quarter_end.astype(int)
        quarter_end_count = df_time['is_quarter_end'].sum()
        print(f"  ✓ is_quarter_end: {quarter_end_count} quarter-end dates")
        
        # Placeholder for days to earnings (to be populated later)
        df_time['days_to_earnings'] = np.nan
        print(f"  ✓ days_to_earnings: placeholder (NaN)")
        
        print(f"\n✓ Temporal features added: 7 new columns")
        
        return df_time
    
    except Exception as e:
        print(f"✗ Error adding temporal features: {e}")
        import traceback
        traceback.print_exc()
        return df


def calculate_daily_return(df):
    """
    Calculate daily return if not already present.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with close prices
    
    Returns:
    --------
    pd.DataFrame
        Dataset with daily_return column
    """
    if 'daily_return' not in df.columns:
        print("\nCalculating daily return...")
        df['daily_return'] = df['close'].pct_change() * 100
        print(f"  ✓ daily_return calculated (percentage)")
    
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


def plot_seasonality_boxplot(df, output_dir):
    """
    Generate seasonality boxplot (close price vs month).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with temporal features
    output_dir : str
        Output directory for the plot
    """
    print("\nGenerating seasonality boxplot...")
    try:
        # Clean data
        plot_df = df[['month', 'close']].dropna().copy()
        
        if len(plot_df) == 0:
            print("✗ No data available for plotting")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Month names for x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create boxplot
        bp = ax.boxplot([plot_df[plot_df['month'] == m]['close'].values 
                         for m in range(1, 13)],
                        labels=month_names,
                        patch_artist=True,
                        notch=True,
                        showfliers=True)
        
        # Customize boxplot colors
        colors = plt.cm.Set3(range(12))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Styling
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Apple Stock Close Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Seasonality Analysis: Apple Stock Price Distribution by Month', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        monthly_means = [plot_df[plot_df['month'] == m]['close'].mean() 
                        for m in range(1, 13)]
        ax.plot(range(1, 13), monthly_means, color='red', linewidth=2, 
               marker='o', markersize=6, label='Monthly Mean', zorder=10)
        ax.legend(loc='upper left', fontsize=11)
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'seasonality_boxplot.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Seasonality boxplot saved to {output_path}")
        plt.close()
    
    except Exception as e:
        print(f"✗ Error generating seasonality boxplot: {e}")
        import traceback
        traceback.print_exc()


def plot_weekday_effect(df, output_dir):
    """
    Generate weekday effect plot (daily return vs day of week).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with temporal features
    output_dir : str
        Output directory for the plot
    """
    print("\nGenerating weekday effect plot...")
    try:
        # Clean data
        plot_df = df[['dow', 'daily_return']].dropna().copy()
        
        if len(plot_df) == 0:
            print("✗ No data available for plotting")
            return
        
        # Create plot with two panels
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Day names for x-axis
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        
        # Panel 1: Boxplot
        ax1 = axes[0]
        bp = ax1.boxplot([plot_df[plot_df['dow'] == d]['daily_return'].values 
                         for d in range(5)],  # 0-4 (Mon-Fri)
                        labels=day_names,
                        patch_artist=True,
                        notch=True,
                        showfliers=True)
        
        # Customize boxplot colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Daily Return (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Daily Return Distribution by Weekday', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Bar chart of mean returns
        ax2 = axes[1]
        daily_means = [plot_df[plot_df['dow'] == d]['daily_return'].mean() 
                      for d in range(5)]
        daily_stds = [plot_df[plot_df['dow'] == d]['daily_return'].std() 
                     for d in range(5)]
        
        bars = ax2.bar(day_names, daily_means, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        ax2.errorbar(day_names, daily_means, yerr=daily_stds, fmt='none', 
                    color='black', capsize=5, linewidth=2, alpha=0.7)
        
        # Color bars based on positive/negative
        for i, (bar, mean) in enumerate(zip(bars, daily_means)):
            if mean < 0:
                bar.set_color('#d62728')
                bar.set_alpha(0.6)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Mean Daily Return (%) ± Std', fontsize=11, fontweight='bold')
        ax2.set_title('Average Daily Return by Weekday (with Std Dev)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, daily_means)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}%',
                    ha='center', va='bottom' if mean >= 0 else 'top',
                    fontsize=9, fontweight='bold')
        
        fig.suptitle('Weekday Effect Analysis: Apple Stock Returns', 
                    fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'weekday_effect.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Weekday effect plot saved to {output_path}")
        plt.close()
    
    except Exception as e:
        print(f"✗ Error generating weekday effect plot: {e}")
        import traceback
        traceback.print_exc()


def print_temporal_statistics(df):
    """
    Print statistics for temporal features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with temporal features
    """
    print("\n" + "="*80)
    print("TEMPORAL FEATURE STATISTICS")
    print("="*80 + "\n")
    
    # Day of week distribution
    print("Day of Week Distribution:")
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i in range(7):
        count = (df['dow'] == i).sum()
        pct = count / len(df) * 100
        print(f"  {dow_names[i]:>10}: {count:>4} ({pct:>5.2f}%)")
    
    print("\nMonth Distribution:")
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    for i in range(1, 13):
        count = (df['month'] == i).sum()
        pct = count / len(df) * 100
        print(f"  {month_names[i-1]:>10}: {count:>4} ({pct:>5.2f}%)")
    
    print("\nQuarter Distribution:")
    for i in range(1, 5):
        count = (df['quarter'] == i).sum()
        pct = count / len(df) * 100
        print(f"  Q{i}: {count:>4} ({pct:>5.2f}%)")
    
    print("\nYear Distribution:")
    for year in sorted(df['year'].unique()):
        count = (df['year'] == year).sum()
        print(f"  {year}: {count:>4} trading days")
    
    print(f"\nSpecial Dates:")
    print(f"  Month-end dates: {df['is_month_end'].sum()}")
    print(f"  Quarter-end dates: {df['is_quarter_end'].sum()}")
    
    # Daily return by weekday
    if 'daily_return' in df.columns:
        print("\nMean Daily Return by Weekday:")
        for i in range(5):  # Monday to Friday
            mean_return = df[df['dow'] == i]['daily_return'].mean()
            print(f"  {dow_names[i]:>10}: {mean_return:>7.4f}%")
        
        # Monthly seasonality
        print("\nMean Close Price by Month:")
        for i in range(1, 13):
            mean_close = df[df['month'] == i]['close'].mean()
            print(f"  {month_names[i-1]:>10}: ${mean_close:>8.2f}")


def main():
    """
    Main execution function:
    1. Load dataset
    2. Add temporal features
    3. Calculate daily return (if needed)
    4. Save dataset
    5. Generate visualizations
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL ANALYSIS - TEMPORAL FEATURES ENGINEERING")
    print("="*80 + "\n")
    
    # Define paths
    input_csv = 'data/processed/apple_feature_plus.csv'
    output_csv = 'data/processed/apple_feature_plus_time.csv'
    figures_dir = 'reports/figures'
    
    # Step 1: Load dataset
    print("[1/5] Loading dataset...")
    df = load_dataset(input_csv)
    if df is None:
        print("✗ Failed to load dataset. Exiting.")
        return False
    
    # Step 2: Add temporal features
    print("\n[2/5] Adding temporal features...")
    df_time = add_temporal_features(df)
    
    # Step 3: Calculate daily return (if needed)
    print("\n[3/5] Ensuring daily return is calculated...")
    df_time = calculate_daily_return(df_time)
    
    # Step 4: Save dataset
    print("\n[4/5] Saving dataset...")
    save_dataset(df_time, output_csv)
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    plot_seasonality_boxplot(df_time, figures_dir)
    plot_weekday_effect(df_time, figures_dir)
    
    # Print statistics
    print_temporal_statistics(df_time)
    
    # Final summary
    print("\n" + "="*80)
    print("TEMPORAL FEATURES ENGINEERING COMPLETE")
    print("="*80)
    
    print(f"\n✓ Output Files:")
    print(f"  • {output_csv}")
    print(f"  • {figures_dir}/seasonality_boxplot.png")
    print(f"  • {figures_dir}/weekday_effect.png")
    
    print(f"\n✓ Temporal Features Added:")
    print(f"  • dow (0-6): Day of week")
    print(f"  • month (1-12): Month")
    print(f"  • quarter (1-4): Quarter")
    print(f"  • year: Year")
    print(f"  • is_month_end: Month-end indicator")
    print(f"  • is_quarter_end: Quarter-end indicator")
    print(f"  • days_to_earnings: Placeholder (NaN)")
    
    print(f"\n✓ Data Quality:")
    print(f"  • Records: {len(df_time):,}")
    print(f"  • Columns: {len(df_time.columns)}")
    print(f"  • Date range: {df_time['date'].min().date()} to {df_time['date'].max().date()}")
    
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
