"""Data loader module for Apple Financial and Social Analysis Dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import time
import yfinance as yf
from pytrends.request import TrendReq

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)


def load_raw_data(file_path):
    """Load raw data from CSV or Excel file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def save_processed_data(df, output_path):
    """Save processed data to CSV file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Data saved to {output_path}")


def ensure_dirs(paths: list):
    """Create any missing directories."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ensured: {path}")


def fetch_yfinance_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL')
    start : str
        Start date in format 'YYYY-MM-DD'
    end : str
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        Stock data with Date, Open, High, Low, Close, Volume, Adj Close
    """
    print(f"Downloading {ticker} stock data from {start} to {end}...")
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns.values]
        
        # Reset index to make Date a column
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        print(f"✓ Downloaded {len(data)} records of {ticker} stock data")
        return data
    except Exception as e:
        print(f"Error downloading {ticker} stock data: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_google_trends(keywords: list, start: str, end: str) -> pd.DataFrame:
    """
    Download Google Trends data for specified keywords with retry logic.
    
    Parameters:
    -----------
    keywords : list
        List of keywords to search for
    start : str
        Start date in format 'YYYY-MM-DD'
    end : str
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        Google Trends data with Date and search interest for each keyword
    """
    print(f"Downloading Google Trends data for {keywords}...")
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(keywords, cat=0, timeframe=f'{start} {end}', geo='US')
            
            print(f"  Fetching data (attempt {attempt + 1}/{max_retries})...")
            trends_data = pytrends.interest_over_time()
            
            # Remove 'isPartial' column if it exists
            if 'isPartial' in trends_data.columns:
                trends_data = trends_data.drop('isPartial', axis=1)
            
            # Reset index to make date a column
            trends_data = trends_data.reset_index()
            trends_data.columns = [col.lower() for col in trends_data.columns]
            
            # Resample to daily frequency with forward fill to handle weekly data
            trends_data['date'] = pd.to_datetime(trends_data['date'])
            trends_data = trends_data.set_index('date')
            trends_data = trends_data.asfreq('D').ffill()
            trends_data = trends_data.reset_index()
            
            print(f"✓ Downloaded {len(trends_data)} records of Google Trends data")
            return trends_data
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Rate limited or error occurred. Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Error downloading Google Trends data after {max_retries} attempts: {e}")
                import traceback
                traceback.print_exc()
                return None


def merge_on_date(price_df: pd.DataFrame, trends_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stock price data with Google Trends data on date.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        Stock price data with 'date' column
    trends_df : pd.DataFrame
        Google Trends data with 'date' column
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset with normalized daily frequency
    """
    print("Merging datasets on date...")
    try:
        # Ensure date columns are datetime
        price_df['date'] = pd.to_datetime(price_df['date']).dt.date
        trends_df['date'] = pd.to_datetime(trends_df['date']).dt.date
        
        # Merge on date
        merged_data = price_df.merge(trends_df, on='date', how='inner')
        
        # Convert date back to datetime and sort
        merged_data['date'] = pd.to_datetime(merged_data['date'])
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
        
        # Drop any remaining NA values
        merged_data = merged_data.dropna()
        
        print(f"✓ Merged dataset contains {len(merged_data)} records")
        print(f"  Columns: {list(merged_data.columns)}")
        return merged_data
    except Exception as e:
        print(f"Error merging datasets: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_plots(merged_df: pd.DataFrame, out_dir: str):
    """
    Generate and save two visualizations from merged dataset.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataset with stock price and trends data
    out_dir : str
        Output directory for PNG files
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Apple Closing Price Time Series
    print("Generating Apple closing price chart...")
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(merged_df['date'], merged_df['close'], linewidth=2, color='#1f77b4', label='AAPL Close Price')
        plt.title('Apple Inc. (AAPL) Stock Closing Price (2015-2025)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Closing Price (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = Path(out_dir) / 'apple_closing_price.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Closing price chart saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating closing price chart: {e}")
    
    # Plot 2: Google Trends Comparison
    print("Generating Google Trends comparison chart...")
    try:
        plt.figure(figsize=(14, 7))
        
        # Find trend columns (exclude date, price columns)
        trend_columns = [col for col in merged_df.columns 
                        if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'adj close']]
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, col in enumerate(trend_columns):
            color = colors[i % len(colors)]
            plt.plot(merged_df['date'], merged_df[col], linewidth=2, label=col.title(), color=color)
        
        plt.title('Google Trends Search Interest Comparison (2015-2025)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Search Interest (0-100)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = Path(out_dir) / 'google_trends_comparison.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Trends comparison chart saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating trends comparison chart: {e}")


def main():
    """
    Main execution function:
    - Ensures all required directories exist
    - Downloads AAPL stock data
    - Fetches Google Trends data
    - Merges and cleans the datasets
    - Saves the merged dataset
    - Generates visualizations
    """
    print("\n" + "="*70)
    print("APPLE FINANCIAL AND SOCIAL ANALYSIS - DATA LOADER")
    print("="*70 + "\n")
    
    # Define paths
    data_raw = 'data/raw'
    data_processed = 'data/processed'
    reports_figures = 'reports/figures'
    
    # Ensure directories exist
    ensure_dirs([data_raw, data_processed, reports_figures])
    
    # Define date range
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    keywords = ['Apple iPhone', 'MacBook']
    
    # Download stock data
    stock_data = fetch_yfinance_data('AAPL', start_date, end_date)
    if stock_data is None:
        print("Failed to download stock data. Exiting.")
        return
    
    # Download Google Trends data
    trends_data = fetch_google_trends(keywords, start_date, end_date)
    if trends_data is None:
        print("Failed to download trends data. Exiting.")
        return
    
    # Merge datasets
    merged_df = merge_on_date(stock_data, trends_data)
    if merged_df is None:
        print("Failed to merge datasets. Exiting.")
        return
    
    # Save merged dataset
    output_csv = Path(data_processed) / 'apple_analysis_dataset.csv'
    save_processed_data(merged_df, str(output_csv))
    
    # Generate visualizations
    make_plots(merged_df, reports_figures)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Date Range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Total Records: {len(merged_df)}")
    print(f"\nClosing Price Statistics (USD):")
    print(f"  Mean:   ${merged_df['close'].mean():.2f}")
    print(f"  Median: ${merged_df['close'].median():.2f}")
    print(f"  Min:    ${merged_df['close'].min():.2f}")
    print(f"  Max:    ${merged_df['close'].max():.2f}")
    print(f"  Std:    ${merged_df['close'].std():.2f}")
    print(f"\nVolume Statistics:")
    print(f"  Mean:   {merged_df['volume'].mean():,.0f}")
    print(f"  Min:    {merged_df['volume'].min():,.0f}")
    print(f"  Max:    {merged_df['volume'].max():,.0f}")
    print(f"\nDataset Shape: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
    print(f"\nColumns: {', '.join(merged_df.columns)}")
    print(f"\nMissing Values:")
    print(merged_df.isnull().sum())
    print("="*70 + "\n")
    
    print("✓ Data loading and processing completed successfully!")


if __name__ == '__main__':
    main()
