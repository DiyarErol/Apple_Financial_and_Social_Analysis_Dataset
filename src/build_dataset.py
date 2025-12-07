"""
Build Dataset Script - Comprehensive data pipeline for Apple Financial and Social Analysis.
Downloads, processes, merges, and visualizes Apple stock data with Google Trends data.
"""

from data_loader import (
    ensure_dirs,
    fetch_yfinance_data,
    fetch_google_trends,
    merge_on_date,
    make_plots,
    save_processed_data
)


def build_dataset():
    """
    Complete data pipeline:
    1. Ensure all required directories exist
    2. Download AAPL stock data (2015-2025)
    3. Fetch Google Trends for Apple iPhone and MacBook
    4. Merge, clean, and save the dataset
    5. Generate visualizations
    """
    print("\n" + "="*80)
    print("BUILDING APPLE FINANCIAL AND SOCIAL ANALYSIS DATASET")
    print("="*80 + "\n")
    
    # Configuration
    config = {
        'paths': {
            'raw': 'data/raw',
            'processed': 'data/processed',
            'figures': 'reports/figures'
        },
        'ticker': 'AAPL',
        'start_date': '2015-01-01',
        'end_date': '2025-12-31',
        'keywords': ['Apple iPhone', 'MacBook'],
        'output_csv': 'data/processed/apple_analysis_dataset.csv'
    }
    
    print(f"Configuration:")
    print(f"  Ticker: {config['ticker']}")
    print(f"  Date Range: {config['start_date']} to {config['end_date']}")
    print(f"  Keywords: {config['keywords']}")
    print(f"  Output CSV: {config['output_csv']}\n")
    
    # Step 1: Ensure directories
    print("[1/5] Ensuring directories exist...")
    paths_list = list(config['paths'].values())
    ensure_dirs(paths_list)
    
    # Step 2: Download stock data
    print("\n[2/5] Downloading AAPL stock data...")
    stock_data = fetch_yfinance_data(
        config['ticker'],
        config['start_date'],
        config['end_date']
    )
    
    if stock_data is None:
        print("✗ Failed to download stock data.")
        return False
    
    print(f"  ✓ Downloaded {len(stock_data):,} records")
    
    # Step 3: Fetch Google Trends
    print("\n[3/5] Fetching Google Trends data...")
    trends_data = fetch_google_trends(
        config['keywords'],
        config['start_date'],
        config['end_date']
    )
    
    if trends_data is None:
        print("✗ Failed to fetch Google Trends data.")
        return False
    
    print(f"  ✓ Downloaded {len(trends_data):,} records")
    
    # Step 4: Merge and save
    print("\n[4/5] Merging and processing datasets...")
    merged_df = merge_on_date(stock_data, trends_data)
    
    if merged_df is None:
        print("✗ Failed to merge datasets.")
        return False
    
    print(f"  ✓ Merged dataset: {len(merged_df):,} records")
    print(f"  ✓ Columns: {', '.join(merged_df.columns)}")
    print(f"  ✓ Missing values: {merged_df.isnull().sum().sum()}")
    
    # Save the dataset
    save_processed_data(merged_df, config['output_csv'])
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    make_plots(merged_df, config['paths']['figures'])
    
    # Print summary
    print("\n" + "="*80)
    print("DATASET BUILD COMPLETE")
    print("="*80)
    
    print(f"\n✓ Summary Statistics:")
    print(f"  Total Records: {len(merged_df):,}")
    print(f"  Date Range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    print(f"  Columns: {len(merged_df.columns)}")
    
    print(f"\n✓ Price Statistics (USD):")
    print(f"  Mean:     ${merged_df['close'].mean():.2f}")
    print(f"  Median:   ${merged_df['close'].median():.2f}")
    print(f"  Std Dev:  ${merged_df['close'].std():.2f}")
    print(f"  Min:      ${merged_df['close'].min():.2f}")
    print(f"  Max:      ${merged_df['close'].max():.2f}")
    
    print(f"\n✓ Volume Statistics:")
    print(f"  Mean:     {merged_df['volume'].mean():,.0f}")
    print(f"  Min:      {merged_df['volume'].min():,.0f}")
    print(f"  Max:      {merged_df['volume'].max():,.0f}")
    
    print(f"\n✓ Output Files Generated:")
    print(f"  • {config['output_csv']}")
    print(f"  • {config['paths']['figures']}/apple_closing_price.png")
    print(f"  • {config['paths']['figures']}/google_trends_comparison.png")
    
    print("\n" + "="*80)
    print("✓ All tasks completed successfully!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = build_dataset()
    exit(0 if success else 1)
