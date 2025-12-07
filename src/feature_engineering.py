"""
Feature Engineering Script for Apple Financial and Social Analysis Dataset.
Creates engineered features for machine learning and predictive analysis.
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
plt.rcParams['figure.figsize'] = (14, 8)


def load_and_prepare_data(csv_path):
    """
    Load macro dataset and prepare for feature engineering.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    
    Returns:
    --------
    pd.DataFrame
        Prepared dataset sorted by date
    """
    print("Loading and preparing data...")
    try:
        df = pd.read_csv(csv_path)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Data loaded: {len(df)} records")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def engineer_features(df):
    """
    Create engineered features for analysis and prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    
    Returns:
    --------
    pd.DataFrame
        Dataset with engineered features
    """
    print("\nEngineering features...")
    try:
        # 1. Daily return
        print("  Creating daily_return...")
        df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # 2. 7-day volatility
        print("  Creating volatility_7d...")
        df['volatility_7d'] = df['close'].pct_change().rolling(window=7).std()
        
        # 3. 7-day moving average
        print("  Creating ma7...")
        df['ma7'] = df['close'].rolling(window=7).mean()
        
        # 4. 30-day moving average
        print("  Creating ma30...")
        df['ma30'] = df['close'].rolling(window=30).mean()
        
        # 5. Trend delta (iPhone vs MacBook)
        print("  Creating trend_delta...")
        df['trend_delta'] = df['apple iphone'] - df['macbook']
        
        # 6. Macro delta (USD vs Fed Rate)
        print("  Creating macro_delta...")
        df['macro_delta'] = df['usd_index'] - df['fed_rate']
        
        # 7. Target variable (next day close price)
        print("  Creating next_day_close (target variable)...")
        df['next_day_close'] = df['close'].shift(-1)
        
        print(f"✓ Features engineered successfully")
        print(f"  New columns: daily_return, volatility_7d, ma7, ma30, trend_delta, macro_delta, next_day_close")
        
        return df
    except Exception as e:
        print(f"✗ Error engineering features: {e}")
        import traceback
        traceback.print_exc()
        return None


def clean_data(df):
    """
    Remove rows with NaN values created by rolling/shifting operations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with engineered features
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    print("\nCleaning data (removing NaN values)...")
    try:
        initial_rows = len(df)
        
        # Drop NaN values
        df = df.dropna()
        
        rows_removed = initial_rows - len(df)
        
        print(f"✓ Data cleaned")
        print(f"  Initial rows: {initial_rows:,}")
        print(f"  Rows removed: {rows_removed:,}")
        print(f"  Final rows: {len(df):,}")
        
        return df
    except Exception as e:
        print(f"✗ Error cleaning data: {e}")
        return df


def save_feature_dataset(df, output_path):
    """
    Save engineered feature dataset to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataset
    output_path : str
        Path to save CSV
    """
    print("Saving feature dataset...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Feature dataset saved to {output_path}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")


def plot_correlation_heatmap_engineered(df, output_dir):
    """
    Generate and save correlation heatmap for engineered features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataset
    output_dir : str
        Output directory
    """
    print("Generating engineered feature correlation heatmap...")
    try:
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   cbar_kws={'label': 'Correlation Coefficient'}, 
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title('Correlation Matrix - Engineered Features', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'correlation_heatmap_engineered.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Engineered correlation heatmap saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating heatmap: {e}")


def plot_feature_distribution(df, output_dir):
    """
    Generate and save distribution plots for engineered features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataset
    output_dir : str
        Output directory
    """
    print("Generating feature distribution plots...")
    try:
        # Engineered features to plot
        engineered_features = ['daily_return', 'volatility_7d', 'ma7', 'ma30', 
                              'trend_delta', 'macro_delta']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(engineered_features):
            ax = axes[idx]
            
            # Histogram with KDE
            df[feature].hist(bins=50, ax=ax, color='#1f77b4', alpha=0.7, edgecolor='black')
            ax_kde = ax.twinx()
            df[feature].plot(kind='kde', ax=ax_kde, color='#d62728', linewidth=2)
            
            ax.set_title(f'Distribution of {feature}', fontweight='bold', fontsize=11)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax_kde.set_ylabel('Density')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'feature_distribution.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Feature distribution plots saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating distribution plots: {e}")


def plot_target_correlation(df, output_dir):
    """
    Generate and save correlation heatmap of all features with target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataset
    output_dir : str
        Output directory
    """
    print("Generating target correlation plot...")
    try:
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation with target variable
        target_corr = numeric_df.corr()['next_day_close'].sort_values(ascending=False)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in target_corr.values]
        target_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.8)
        
        ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Correlation with Target Variable (Next Day Close Price)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'target_correlation.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Target correlation plot saved to {output_path}")
        plt.close()
        
        # Print correlation values
        print("\n  Feature Correlations with Next Day Close:")
        print("  " + "="*50)
        for feature, corr_val in target_corr.items():
            if pd.notna(corr_val):
                print(f"  {feature:20s}: {corr_val:7.4f}")
        print("  " + "="*50)
    except Exception as e:
        print(f"✗ Error generating target correlation plot: {e}")


def print_feature_statistics(df):
    """
    Print detailed statistics for engineered features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataset
    """
    print("\n" + "="*80)
    print("ENGINEERED FEATURES STATISTICS")
    print("="*80)
    
    engineered_features = ['daily_return', 'volatility_7d', 'ma7', 'ma30', 
                          'trend_delta', 'macro_delta', 'next_day_close']
    
    for feature in engineered_features:
        if feature in df.columns:
            print(f"\n{feature}:")
            print(f"  Mean:     {df[feature].mean():10.4f}")
            print(f"  Median:   {df[feature].median():10.4f}")
            print(f"  Std:      {df[feature].std():10.4f}")
            print(f"  Min:      {df[feature].min():10.4f}")
            print(f"  Max:      {df[feature].max():10.4f}")
    
    print("\n" + "="*80)


def main():
    """
    Main execution function:
    1. Load and prepare data
    2. Engineer features
    3. Clean data
    4. Save feature dataset
    5. Generate visualizations
    6. Print statistics
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL AND SOCIAL ANALYSIS - FEATURE ENGINEERING")
    print("="*80 + "\n")
    
    # Define paths
    input_csv = 'data/processed/apple_macro_dataset.csv'
    output_csv = 'data/processed/apple_feature_dataset.csv'
    output_figures_dir = 'reports/figures'
    
    # Step 1: Load and prepare data
    print("[1/6] Loading and preparing data...")
    df = load_and_prepare_data(input_csv)
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return False
    
    # Step 2: Engineer features
    print("\n[2/6] Engineering features...")
    df = engineer_features(df)
    if df is None:
        print("✗ Failed to engineer features. Exiting.")
        return False
    
    # Step 3: Clean data
    print("\n[3/6] Cleaning data...")
    df = clean_data(df)
    
    # Step 4: Save feature dataset
    print("\n[4/6] Saving feature dataset...")
    save_feature_dataset(df, output_csv)
    
    # Step 5: Generate visualizations
    print("\n[5/6] Generating visualizations...")
    plot_correlation_heatmap_engineered(df, output_figures_dir)
    plot_feature_distribution(df, output_figures_dir)
    plot_target_correlation(df, output_figures_dir)
    
    # Step 6: Print statistics
    print("\n[6/6] Computing statistics...")
    print_feature_statistics(df)
    
    # Summary
    print("="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    
    print(f"\n✓ Engineered Features Created:")
    print(f"  • daily_return: Percentage change in closing price")
    print(f"  • volatility_7d: 7-day rolling standard deviation of returns")
    print(f"  • ma7: 7-day moving average of closing price")
    print(f"  • ma30: 30-day moving average of closing price")
    print(f"  • trend_delta: Difference between Apple iPhone and MacBook trends")
    print(f"  • macro_delta: Difference between USD Index and Federal Funds Rate")
    print(f"  • next_day_close: Target variable (next day closing price)")
    
    print(f"\n✓ Output Files:")
    print(f"  • {output_csv}")
    print(f"  • {output_figures_dir}/correlation_heatmap_engineered.png")
    print(f"  • {output_figures_dir}/feature_distribution.png")
    print(f"  • {output_figures_dir}/target_correlation.png")
    
    print(f"\n✓ Final Dataset:")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    print("\n" + "="*80)
    print("✓ All feature engineering tasks completed successfully!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
