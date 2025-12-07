"""Analysis script for Apple Financial and Social Analysis Dataset.
Performs correlation analysis and generates comprehensive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)


def load_and_prepare_data(csv_path):
    """
    Load CSV data and prepare it for analysis.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index
    """
    print("Loading data...")
    try:
        df = pd.read_csv(csv_path)
        
        # Convert date column to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print(f"✓ Data loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_moving_averages(df):
    """
    Calculate moving averages for closing price.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new moving average columns
    """
    print("\nCalculating moving averages...")
    try:
        df['close_MA30'] = df['close'].rolling(window=30).mean()
        
        print(f"✓ 30-day moving average calculated")
        print(f"  First MA value: {df['close_MA30'].dropna().iloc[0]:.2f}")
        print(f"  Last MA value:  {df['close_MA30'].dropna().iloc[-1]:.2f}")
        
        return df
    except Exception as e:
        print(f"✗ Error calculating moving averages: {e}")
        return df


def calculate_correlations(df):
    """
    Calculate Pearson correlation between key columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    print("\nCalculating correlations...")
    try:
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove moving average for correlation analysis
        if 'close_MA30' in numeric_cols:
            numeric_cols.remove('close_MA30')
        
        correlation_matrix = df[numeric_cols].corr(method='pearson')
        
        print("✓ Pearson correlation matrix calculated:")
        print("\n" + "="*80)
        print("CORRELATION MATRIX")
        print("="*80)
        print(correlation_matrix.to_string())
        print("="*80 + "\n")
        
        return correlation_matrix
    except Exception as e:
        print(f"✗ Error calculating correlations: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_correlation_matrix(corr_matrix, output_path):
    """
    Save correlation matrix to CSV.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    output_path : str
        Path to save the CSV
    """
    print("Saving correlation matrix...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        corr_matrix.to_csv(output_path)
        print(f"✓ Correlation matrix saved to {output_path}")
    except Exception as e:
        print(f"✗ Error saving correlation matrix: {e}")


def plot_correlation_heatmap(df, output_dir):
    """
    Generate and save correlation heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    output_dir : str
        Output directory for the figure
    """
    print("Generating correlation heatmap...")
    try:
        # Select numeric columns for heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove moving average for cleaner heatmap
        if 'close_MA30' in numeric_cols:
            numeric_cols.remove('close_MA30')
        
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   cbar_kws={'label': 'Correlation Coefficient'}, 
                   square=True, linewidths=1, fmt='.3f')
        plt.title('Correlation Matrix - Apple Stock and Google Trends', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'correlation_heatmap.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating heatmap: {e}")


def plot_price_vs_iphone_trend(df, output_dir):
    """
    Generate and save dual-axis plot of closing price vs. iPhone trend.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    output_dir : str
        Output directory for the figure
    """
    print("Generating price vs. iPhone trend plot...")
    try:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plot closing price on primary axis
        color1 = '#1f77b4'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Closing Price (USD)', fontsize=12, color=color1)
        line1 = ax1.plot(df.index, df['close'], color=color1, linewidth=2, label='Closing Price')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot iPhone trend on secondary axis
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        ax2.set_ylabel('Google Trends - Apple iPhone (0-100)', fontsize=12, color=color2)
        line2 = ax2.plot(df.index, df['apple iphone'], color=color2, linewidth=2, 
                        label='Apple iPhone Trend')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Title and legend
        plt.title('Apple Stock Closing Price vs. Apple iPhone Google Trends', 
                 fontsize=14, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=11)
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'price_vs_iphone_trend.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Price vs. iPhone trend plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating price vs. iPhone plot: {e}")


def plot_macbook_vs_price_scatter(df, output_dir):
    """
    Generate and save scatter plot comparing MacBook trend vs. closing price.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    output_dir : str
        Output directory for the figure
    """
    print("Generating MacBook trend vs. price scatter plot...")
    try:
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with color gradient based on date
        scatter = plt.scatter(df['macbook'], df['close'], 
                            c=range(len(df)), cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add colorbar to show time progression
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time Progression (Earlier → Later)', fontsize=11)
        
        # Labels and title
        plt.xlabel('Google Trends - MacBook (0-100)', fontsize=12)
        plt.ylabel('Apple Stock Closing Price (USD)', fontsize=12)
        plt.title('MacBook Google Trends vs. Apple Stock Closing Price', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient to plot
        corr = df['macbook'].corr(df['close'])
        plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}', 
                transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'macbook_vs_price_scatter.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ MacBook vs. price scatter plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating scatter plot: {e}")


def main():
    """
    Main execution function:
    1. Load and prepare data
    2. Calculate moving averages
    3. Calculate correlations
    4. Save correlation matrix
    5. Generate visualizations
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL AND SOCIAL ANALYSIS - CORRELATION AND ANALYSIS")
    print("="*80 + "\n")
    
    # Define paths
    input_csv = 'data/processed/apple_analysis_dataset.csv'
    output_figures_dir = 'reports/figures'
    output_final_dir = 'reports/final'
    output_corr_csv = 'reports/final/correlation_matrix.csv'
    
    # Load data
    df = load_and_prepare_data(input_csv)
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return False
    
    # Calculate moving averages
    df = calculate_moving_averages(df)
    
    # Calculate correlations
    corr_matrix = calculate_correlations(df)
    if corr_matrix is None:
        print("✗ Failed to calculate correlations. Exiting.")
        return False
    
    # Save correlation matrix
    save_correlation_matrix(corr_matrix, output_corr_csv)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_correlation_heatmap(df, output_figures_dir)
    plot_price_vs_iphone_trend(df, output_figures_dir)
    plot_macbook_vs_price_scatter(df, output_figures_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\n✓ Output Files Generated:")
    print(f"  • {output_corr_csv}")
    print(f"  • {output_figures_dir}/correlation_heatmap.png")
    print(f"  • {output_figures_dir}/price_vs_iphone_trend.png")
    print(f"  • {output_figures_dir}/macbook_vs_price_scatter.png")
    
    print(f"\n✓ Key Statistics:")
    print(f"  Dataset Records: {len(df):,}")
    print(f"  Moving Average Records: {df['close_MA30'].notna().sum():,}")
    
    print("\n" + "="*80)
    print("✓ All analysis tasks completed successfully!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
