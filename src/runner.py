"""
Runner script for Apple Financial and Social Analysis Dataset.
Imports and orchestrates all data loading, processing, and visualization functions.
"""

from data_loader import (
    ensure_dirs,
    fetch_yfinance_data,
    fetch_google_trends,
    merge_on_date,
    make_plots,
    save_processed_data
)


def main():
    """
    Main execution function that orchestrates the entire data pipeline:
    1. Ensure all required directories exist
    2. Download AAPL stock data
    3. Fetch Google Trends data
    4. Merge and clean the datasets
    5. Save the merged dataset to CSV
    6. Generate visualizations
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL AND SOCIAL ANALYSIS - RUNNER")
    print("="*80 + "\n")
    
    # Step 1: Define paths
    print("[Step 1] Defining paths...")
    paths = [
        'data/raw',
        'data/processed',
        'reports/figures'
    ]
    
    # Step 2: Ensure directories exist
    print("[Step 2] Ensuring directories exist...")
    ensure_dirs(paths)
    
    # Step 3: Download AAPL stock data
    print("\n[Step 3] Downloading AAPL stock data...")
    start_date = '2015-01-01'
    end_date = '2025-12-31'
    stock_data = fetch_yfinance_data('AAPL', start_date, end_date)
    
    if stock_data is None:
        print("✗ Failed to download stock data. Exiting.")
        return
    
    # Step 4: Fetch Google Trends data
    print("\n[Step 4] Fetching Google Trends data...")
    keywords = ['Apple iPhone', 'MacBook']
    trends_data = fetch_google_trends(keywords, start_date, end_date)
    
    if trends_data is None:
        print("✗ Failed to fetch Google Trends data. Exiting.")
        return
    
    # Step 5: Merge datasets on date
    print("\n[Step 5] Merging datasets on date...")
    merged_df = merge_on_date(stock_data, trends_data)
    
    if merged_df is None:
        print("✗ Failed to merge datasets. Exiting.")
        return
    
    # Step 6: Save merged dataset to CSV
    print("\n[Step 6] Saving merged dataset...")
    output_csv_path = 'data/processed/apple_analysis_dataset.csv'
    save_processed_data(merged_df, output_csv_path)
    
    # Step 7: Generate visualizations
    print("\n[Step 7] Generating visualizations...")
    output_figures_dir = 'reports/figures'
    make_plots(merged_df, output_figures_dir)
    
    # Step 8: Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"\n✓ Stock Data:")
    print(f"  Records: {len(stock_data):,}")
    print(f"  Date Range: {stock_data['date'].min()} to {stock_data['date'].max()}")
    
    print(f"\n✓ Google Trends Data:")
    print(f"  Records: {len(trends_data):,}")
    print(f"  Date Range: {trends_data['date'].min()} to {trends_data['date'].max()}")
    
    print(f"\n✓ Merged Dataset:")
    print(f"  Records: {len(merged_df):,}")
    print(f"  Columns: {len(merged_df.columns)}")
    print(f"  Date Range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"  Missing Values: {merged_df.isnull().sum().sum()}")
    
    print(f"\n✓ Output Files:")
    print(f"  CSV: {output_csv_path}")
    print(f"  Figures: {output_figures_dir}/apple_closing_price.png")
    print(f"  Figures: {output_figures_dir}/google_trends_comparison.png")
    
    print("\n✓ Price Statistics (USD):")
    print(f"  Mean:   ${merged_df['close'].mean():.2f}")
    print(f"  Median: ${merged_df['close'].median():.2f}")
    print(f"  Min:    ${merged_df['close'].min():.2f}")
    print(f"  Max:    ${merged_df['close'].max():.2f}")
    print(f"  Std:    ${merged_df['close'].std():.2f}")
    
    print("\n✓ Volume Statistics:")
    print(f"  Mean:   {merged_df['volume'].mean():,.0f}")
    print(f"  Min:    {merged_df['volume'].min():,.0f}")
    print(f"  Max:    {merged_df['volume'].max():,.0f}")
    
    print(f"\n✓ Dataset Columns:")
    for i, col in enumerate(merged_df.columns, 1):
        print(f"  {i}. {col}")
    
    print("\n" + "="*80)
    print("✓ Pipeline execution completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
