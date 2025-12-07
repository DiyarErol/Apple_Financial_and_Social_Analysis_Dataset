"""
Script to download Apple stock data and Google Trends data, merge them, and generate visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import yfinance as yf
from pytrends.request import TrendReq

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)


def download_apple_stock_data(start_date='2015-01-01', end_date='2025-12-31'):
    """
    Download Apple's historical stock data from Yahoo Finance.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        Apple stock data with date, Open, High, Low, Close, Volume
    """
    print("Downloading Apple stock data from yfinance...")
    try:
        apple_data = yf.download('AAPL', start=start_date, end=end_date, progress=False)
        
        # Flatten column names (MultiIndex columns)
        if isinstance(apple_data.columns, pd.MultiIndex):
            apple_data.columns = [col[0] for col in apple_data.columns.values]
        
        # Reset index to make date a column
        apple_data = apple_data.reset_index()
        apple_data.columns = [col.lower() for col in apple_data.columns]
        
        print(f"✓ Downloaded {len(apple_data)} records of Apple stock data")
        print(f"  Columns: {list(apple_data.columns)}")
        return apple_data
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_google_trends_data(keywords, start_date='2015-01-01', end_date='2025-12-31'):
    """
    Download Google Trends data for specified keywords.
    
    Parameters:
    -----------
    keywords : list
        List of keywords to search for (e.g., ['Apple iPhone', 'MacBook'])
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        Google Trends data with date and search interest for each keyword
    """
    print("Downloading Google Trends data...")
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Build payload and get interest over time
        pytrends.build_payload(keywords, cat=0, timeframe=f'{start_date} {end_date}', geo='US')
        trends_data = pytrends.interest_over_time()
        
        # Remove 'isPartial' column if it exists
        if 'isPartial' in trends_data.columns:
            trends_data = trends_data.drop('isPartial', axis=1)
        
        # Reset index to make date a column
        trends_data = trends_data.reset_index()
        trends_data.columns = [col.lower() for col in trends_data.columns]
        
        print(f"✓ Downloaded {len(trends_data)} records of Google Trends data")
        print(f"  Columns: {list(trends_data.columns)}")
        return trends_data
    except Exception as e:
        print(f"Error downloading trends data: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_datasets(stock_data, trends_data):
    """
    Merge stock data and trends data on date.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Apple stock data
    trends_data : pd.DataFrame
        Google Trends data
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("Merging datasets on date...")
    try:
        # Ensure date columns are in the same format
        stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date
        trends_data['date'] = pd.to_datetime(trends_data['date']).dt.date
        
        merged_data = stock_data.merge(trends_data, on='date', how='inner')
        print(f"✓ Merged dataset contains {len(merged_data)} records")
        print(f"Columns: {list(merged_data.columns)}")
        return merged_data
    except Exception as e:
        print(f"Error merging datasets: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_dataset(df, output_path):
    """
    Save the merged dataset to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to save
    output_path : str
        Path to save the CSV file
    """
    print(f"Saving dataset to {output_path}...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset saved successfully ({len(df)} rows, {len(df.columns)} columns)")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def plot_closing_price(df, output_path):
    """
    Plot Apple's closing price over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset containing 'date' and 'Close' columns
    output_path : str
        Path to save the figure
    """
    print("Generating closing price plot...")
    try:
        plt.figure(figsize=(14, 7))
        
        # Convert date to datetime for plotting
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values('date')
        
        plt.plot(df_sorted['date'], df_sorted['close'], linewidth=2, color='#1f77b4', label='AAPL Close Price')
        plt.title('Apple Inc. (AAPL) Stock Closing Price (2015-2025)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Closing Price (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Closing price plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating closing price plot: {e}")


def plot_google_trends_comparison(df, keywords, output_path):
    """
    Plot Google Trends comparison for specified keywords.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset
    keywords : list
        Keywords that were used in Google Trends (e.g., ['Apple iPhone', 'MacBook'])
    output_path : str
        Path to save the figure
    """
    print("Generating Google Trends comparison plot...")
    try:
        plt.figure(figsize=(14, 7))
        
        # Convert date to datetime for plotting
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values('date')
        
        colors = ['#ff7f0e', '#2ca02c']
        
        for i, keyword in enumerate(keywords):
            if keyword in df_sorted.columns:
                plt.plot(df_sorted['date'], df_sorted[keyword], 
                        linewidth=2, label=keyword, color=colors[i])
        
        plt.title('Google Trends Search Interest Comparison (2015-2025)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Search Interest (0-100)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Google Trends comparison plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating trends comparison plot: {e}")


def print_summary_statistics(df):
    """
    Print summary statistics of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    print(f"\nDate Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total Records: {len(df)}")
    print(f"\nClosing Price Statistics (USD):")
    print(f"  Mean:   ${df['close'].mean():.2f}")
    print(f"  Median: ${df['close'].median():.2f}")
    print(f"  Min:    ${df['close'].min():.2f}")
    print(f"  Max:    ${df['close'].max():.2f}")
    print(f"  Std:    ${df['close'].std():.2f}")
    print(f"\nVolume Statistics:")
    print(f"  Mean:   {df['volume'].mean():.0f}")
    print(f"  Min:    {df['volume'].min():.0f}")
    print(f"  Max:    {df['volume'].max():.0f}")
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print("="*60 + "\n")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("APPLE FINANCIAL AND SOCIAL ANALYSIS DATASET")
    print("="*60 + "\n")
    
    # Date range
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    keywords = ['Apple iPhone', 'MacBook']
    
    # Download data
    apple_stock = download_apple_stock_data(start_date, end_date)
    if apple_stock is None:
        print("Failed to download stock data. Exiting.")
        return
    
    google_trends = download_google_trends_data(keywords, start_date, end_date)
    if google_trends is None:
        print("Failed to download trends data. Exiting.")
        return
    
    # Merge datasets
    merged_df = merge_datasets(apple_stock, google_trends)
    if merged_df is None:
        print("Failed to merge datasets. Exiting.")
        return
    
    # Save merged dataset
    output_csv = r'data/processed/apple_analysis_dataset.csv'
    save_dataset(merged_df, output_csv)
    
    # Generate plots
    output_price_plot = r'reports/figures/apple_closing_price.png'
    plot_closing_price(merged_df, output_price_plot)
    
    output_trends_plot = r'reports/figures/google_trends_comparison.png'
    plot_google_trends_comparison(merged_df, keywords, output_trends_plot)
    
    # Print summary statistics
    print_summary_statistics(merged_df)
    
    print("✓ Script execution completed successfully!")


if __name__ == '__main__':
    main()
