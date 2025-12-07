"""
Earnings Events Integration Script for Apple Financial and Social Analysis Dataset.
Fetches AAPL earnings calendar and integrates earnings-related features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import requests
import os

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['figure.dpi'] = 300


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_dataset(csv_path):
    """
    Load the feature-engineered dataset.
    
    Parameters:
    -----------
    csv_path : Path
        Path to apple_feature_plus_time_lag.csv
    
    Returns:
    --------
    pd.DataFrame
        Dataset with datetime index
    """
    print("\n" + "="*80)
    print("[1/6] LOADING DATASET")
    print("="*80)
    
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Dataset loaded: {len(df):,} records, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# 2. EARNINGS DATA FETCHING
# ============================================================================

def fetch_earnings_alpha_vantage(symbol='AAPL'):
    """
    Fetch earnings data from Alpha Vantage API.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (default: AAPL)
    
    Returns:
    --------
    pd.DataFrame or None
        Earnings data with columns: reportedDate, reportedEPS, estimatedEPS, 
        surprise, surprisePercentage
    """
    print("\nAttempting to fetch earnings from Alpha Vantage...")
    
    api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
    
    if not api_key:
        print("  ⚠ ALPHAVANTAGE_API_KEY not found in environment variables")
        return None
    
    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': api_key
        }
        
        print(f"  Requesting data from Alpha Vantage for {symbol}...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            print(f"  ✗ API Error: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            print(f"  ⚠ API Limit: {data['Note']}")
            return None
        
        if 'quarterlyEarnings' not in data:
            print("  ✗ No 'quarterlyEarnings' field in response")
            return None
        
        # Parse quarterly earnings
        quarterly = data['quarterlyEarnings']
        
        if not quarterly:
            print("  ✗ Empty quarterlyEarnings data")
            return None
        
        # Convert to DataFrame
        df_earnings = pd.DataFrame(quarterly)
        
        # Select and rename columns
        required_cols = ['reportedDate', 'reportedEPS', 'estimatedEPS', 
                        'surprise', 'surprisePercentage']
        
        # Check which columns exist
        available_cols = [col for col in required_cols if col in df_earnings.columns]
        
        if 'reportedDate' not in available_cols:
            print("  ✗ 'reportedDate' not found in response")
            return None
        
        df_earnings = df_earnings[available_cols].copy()
        
        # Convert date column
        df_earnings['reportedDate'] = pd.to_datetime(df_earnings['reportedDate'])
        
        # Convert numeric columns
        numeric_cols = ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage']
        for col in numeric_cols:
            if col in df_earnings.columns:
                df_earnings[col] = pd.to_numeric(df_earnings[col], errors='coerce')
        
        # Sort by date
        df_earnings = df_earnings.sort_values('reportedDate').reset_index(drop=True)
        
        print(f"  ✓ Alpha Vantage data fetched: {len(df_earnings)} earnings records")
        print(f"    Date range: {df_earnings['reportedDate'].min().date()} to "
              f"{df_earnings['reportedDate'].max().date()}")
        
        return df_earnings
    
    except requests.exceptions.Timeout:
        print("  ✗ Request timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request error: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error fetching from Alpha Vantage: {e}")
        return None


def fetch_earnings_yfinance(symbol='AAPL'):
    """
    Fetch earnings data from yfinance (fallback method).
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (default: AAPL)
    
    Returns:
    --------
    pd.DataFrame or None
        Earnings data with date column
    """
    print("\nAttempting to fetch earnings from yfinance...")
    
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        
        # Build earnings DataFrame from multiple sources
        earnings_data = []
        
        # Method 1: Try earnings_dates attribute (most reliable for historical dates)
        try:
            earnings_dates_df = ticker.earnings_dates
            if earnings_dates_df is not None and not earnings_dates_df.empty:
                print(f"  Found earnings_dates: {len(earnings_dates_df)} records")
                for date, row in earnings_dates_df.iterrows():
                    eps_estimate = row.get('EPS Estimate', np.nan)
                    eps_reported = row.get('Reported EPS', np.nan)
                    
                    # Calculate surprise if both available
                    surprise = np.nan
                    surprise_pct = np.nan
                    if pd.notna(eps_reported) and pd.notna(eps_estimate) and eps_estimate != 0:
                        surprise = eps_reported - eps_estimate
                        surprise_pct = (surprise / abs(eps_estimate)) * 100
                    
                    earnings_data.append({
                        'reportedDate': pd.to_datetime(date),
                        'reportedEPS': eps_reported,
                        'estimatedEPS': eps_estimate,
                        'surprise': surprise,
                        'surprisePercentage': surprise_pct
                    })
        except Exception as e:
            print(f"  ⚠ Could not fetch earnings_dates: {e}")
        
        # Method 2: Try quarterly_earnings (backup for dates)
        if not earnings_data:
            try:
                quarterly_earnings = ticker.quarterly_earnings
                if quarterly_earnings is not None and not quarterly_earnings.empty:
                    print(f"  Found quarterly_earnings: {len(quarterly_earnings)} records")
                    for date, row in quarterly_earnings.iterrows():
                        earnings_data.append({
                            'reportedDate': pd.to_datetime(date),
                            'reportedEPS': row.get('Earnings', np.nan)
                        })
            except Exception as e:
                print(f"  ⚠ Could not fetch quarterly_earnings: {e}")
        
        # Method 3: Try calendar for future dates
        if not earnings_data:
            try:
                calendar = ticker.calendar
                if calendar is not None and 'Earnings Date' in calendar:
                    earnings_dates = calendar['Earnings Date']
                    if isinstance(earnings_dates, pd.Series):
                        earnings_dates = earnings_dates.tolist()
                    elif not isinstance(earnings_dates, list):
                        earnings_dates = [earnings_dates]
                    
                    for date in earnings_dates:
                        earnings_data.append({
                            'reportedDate': pd.to_datetime(date),
                            'reportedEPS': np.nan
                        })
            except Exception as e:
                print(f"  ⚠ Could not fetch calendar: {e}")
        
        if not earnings_data:
            print("  ✗ No earnings data available from yfinance")
            return None
        
        df_earnings = pd.DataFrame(earnings_data)
        df_earnings = df_earnings.drop_duplicates(subset=['reportedDate'])
        df_earnings = df_earnings.sort_values('reportedDate').reset_index(drop=True)
        
        print(f"  ✓ yfinance data fetched: {len(df_earnings)} earnings records")
        if len(df_earnings) > 0:
            print(f"    Date range: {df_earnings['reportedDate'].min().date()} to "
                  f"{df_earnings['reportedDate'].max().date()}")
        
        return df_earnings
    
    except ImportError:
        print("  ✗ yfinance not installed")
        return None
    except Exception as e:
        print(f"  ✗ Error fetching from yfinance: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_earnings_data(symbol='AAPL'):
    """
    Fetch earnings data with fallback strategy.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    
    Returns:
    --------
    pd.DataFrame or None
        Earnings data
    """
    print("\n" + "="*80)
    print("[2/6] FETCHING EARNINGS DATA")
    print("="*80)
    
    # Try Alpha Vantage first
    df_earnings = fetch_earnings_alpha_vantage(symbol)
    
    # Fallback to yfinance
    if df_earnings is None:
        print("\n⚠ Alpha Vantage failed, falling back to yfinance...")
        df_earnings = fetch_earnings_yfinance(symbol)
    
    if df_earnings is None:
        print("\n✗ Failed to fetch earnings data from all sources")
        return None
    
    # Normalize dates to UTC
    df_earnings['reportedDate'] = pd.to_datetime(df_earnings['reportedDate']).dt.normalize()
    
    # Filter to relevant date range (2015-2025)
    df_earnings = df_earnings[
        (df_earnings['reportedDate'] >= '2015-01-01') & 
        (df_earnings['reportedDate'] <= '2025-12-31')
    ].copy()
    
    print(f"\n✓ Final earnings data: {len(df_earnings)} records")
    print(f"  Date range: {df_earnings['reportedDate'].min().date()} to "
          f"{df_earnings['reportedDate'].max().date()}")
    
    return df_earnings


# ============================================================================
# 3. EARNINGS FEATURE ENGINEERING
# ============================================================================

def engineer_earnings_features(df, df_earnings):
    """
    Add earnings-related features to the main dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Main dataset with 'date' column
    df_earnings : pd.DataFrame
        Earnings data with 'reportedDate' column
    
    Returns:
    --------
    pd.DataFrame
        Dataset with earnings features
    """
    print("\n" + "="*80)
    print("[3/6] ENGINEERING EARNINGS FEATURES")
    print("="*80)
    
    try:
        # Ensure dates are normalized
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        
        earnings_dates = df_earnings['reportedDate'].dt.normalize().sort_values().values
        
        print(f"\nProcessing {len(earnings_dates)} earnings dates...")
        
        # 1. is_earnings_day
        print("  Computing is_earnings_day...")
        df['is_earnings_day'] = df['date'].isin(earnings_dates).astype(int)
        print(f"    ✓ Found {df['is_earnings_day'].sum()} earnings days in dataset")
        
        # 2. days_to_earnings (days until next earnings)
        print("  Computing days_to_earnings...")
        days_to_earnings = []
        
        for date in df['date']:
            future_earnings = earnings_dates[earnings_dates > date]
            if len(future_earnings) > 0:
                days_diff = (pd.Timestamp(future_earnings[0]) - pd.Timestamp(date)).days
                # Cap at 90 days
                days_to_earnings.append(min(days_diff, 90))
            else:
                days_to_earnings.append(np.nan)
        
        df['days_to_earnings'] = days_to_earnings
        print(f"    ✓ days_to_earnings: {df['days_to_earnings'].notna().sum()} non-null values")
        
        # 3. days_since_earnings (days since last earnings)
        print("  Computing days_since_earnings...")
        days_since_earnings = []
        
        for date in df['date']:
            past_earnings = earnings_dates[earnings_dates < date]
            if len(past_earnings) > 0:
                days_diff = (pd.Timestamp(date) - pd.Timestamp(past_earnings[-1])).days
                # Cap at 90 days
                days_since_earnings.append(min(days_diff, 90))
            else:
                days_since_earnings.append(np.nan)
        
        df['days_since_earnings'] = days_since_earnings
        print(f"    ✓ days_since_earnings: {df['days_since_earnings'].notna().sum()} non-null values")
        
        # 4. earnings_surprise and earnings_surprise_pct
        print("  Computing earnings surprise metrics...")
        
        # Create mapping from date to surprise metrics
        surprise_map = {}
        surprise_pct_map = {}
        
        if 'surprise' in df_earnings.columns:
            for _, row in df_earnings.iterrows():
                date = row['reportedDate']
                surprise_map[date] = row['surprise']
        
        if 'surprisePercentage' in df_earnings.columns:
            for _, row in df_earnings.iterrows():
                date = row['reportedDate']
                surprise_pct_map[date] = row['surprisePercentage']
        
        # Map to main dataset
        df['earnings_surprise'] = df['date'].map(surprise_map)
        df['earnings_surprise_pct'] = df['date'].map(surprise_pct_map)
        
        print(f"    ✓ earnings_surprise: {df['earnings_surprise'].notna().sum()} non-null values")
        print(f"    ✓ earnings_surprise_pct: {df['earnings_surprise_pct'].notna().sum()} non-null values")
        
        # 5. revenue_surprise (optional, set as NaN for now)
        print("  Setting revenue_surprise as NaN (Alpha Vantage INCOME_STATEMENT not implemented)...")
        df['revenue_surprise'] = np.nan
        
        # Summary
        print("\n✓ Earnings features added:")
        print(f"  - is_earnings_day: {df['is_earnings_day'].sum()} earnings days")
        print(f"  - days_to_earnings: mean={df['days_to_earnings'].mean():.1f} days")
        print(f"  - days_since_earnings: mean={df['days_since_earnings'].mean():.1f} days")
        print(f"  - earnings_surprise: {df['earnings_surprise'].notna().sum()} values")
        print(f"  - earnings_surprise_pct: {df['earnings_surprise_pct'].notna().sum()} values")
        print(f"  - revenue_surprise: placeholder (all NaN)")
        
        return df
    
    except Exception as e:
        print(f"\n✗ Error engineering earnings features: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# 4. DATA SAVING
# ============================================================================

def save_dataset(df, output_path):
    """
    Save the dataset with earnings features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to save
    output_path : Path
        Output file path
    """
    print("\n" + "="*80)
    print("[4/6] SAVING DATASET")
    print("="*80)
    
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        df.to_csv(output_path, index=False)
        
        file_size = output_path.stat().st_size / 1024  # KB
        
        print(f"✓ Dataset saved: {output_path}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  File size: {file_size:.1f} KB")
        
        # Show new columns
        new_cols = ['is_earnings_day', 'days_to_earnings', 'days_since_earnings',
                   'earnings_surprise', 'earnings_surprise_pct', 'revenue_surprise']
        print(f"\n  New columns added: {new_cols}")
        
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_earnings_impact(df, output_path):
    """
    Plot earnings impact visualization.
    
    Dual-axis plot:
    - Primary axis: Close price over time
    - Secondary axis: Earnings surprise percentage (points on earnings days)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with earnings features
    output_path : Path
        Output file path for figure
    """
    print("\n" + "="*80)
    print("[5/6] GENERATING VISUALIZATION")
    print("="*80)
    
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter data with earnings
        df_earnings_days = df[df['is_earnings_day'] == 1].copy()
        
        print(f"\nPlotting earnings impact...")
        print(f"  Total data points: {len(df):,}")
        print(f"  Earnings days: {len(df_earnings_days)}")
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # Primary axis: Close price
        ax1.plot(df['date'], df['close'], linewidth=1.5, color='#1f77b4', 
                label='Close Price', alpha=0.8)
        
        # Mark earnings days with vertical lines
        for date in df_earnings_days['date']:
            ax1.axvline(x=date, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Close Price (USD)', fontsize=12, fontweight='bold', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis: Earnings surprise percentage
        ax2 = ax1.twinx()
        
        # Plot earnings surprise on earnings days only
        earnings_with_surprise = df_earnings_days[df_earnings_days['earnings_surprise_pct'].notna()]
        
        if len(earnings_with_surprise) > 0:
            # Scatter plot for earnings surprise
            scatter = ax2.scatter(earnings_with_surprise['date'], 
                                 earnings_with_surprise['earnings_surprise_pct'],
                                 c=earnings_with_surprise['earnings_surprise_pct'],
                                 cmap='RdYlGn', s=100, alpha=0.8, 
                                 edgecolors='black', linewidths=1,
                                 label='Earnings Surprise %')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
            cbar.set_label('Earnings Surprise %', fontsize=10, fontweight='bold')
            
            ax2.set_ylabel('Earnings Surprise (%)', fontsize=12, fontweight='bold', color='#d62728')
            ax2.tick_params(axis='y', labelcolor='#d62728')
            
            # Add horizontal line at 0
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            print(f"  Plotted {len(earnings_with_surprise)} earnings surprises")
        else:
            ax2.set_ylabel('Earnings Surprise (%) - No Data', fontsize=12, 
                          fontweight='bold', color='gray')
            print("  ⚠ No earnings surprise data to plot")
        
        # Title and legend
        plt.title('Apple Stock Price and Earnings Impact\n'
                 'Vertical dashed lines indicate earnings announcement dates',
                 fontsize=14, fontweight='bold', pad=20)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if len(earnings_with_surprise) > 0:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                      fontsize=10, framealpha=0.9)
        else:
            ax1.legend(lines1, labels1, loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualization saved: {output_path}")
        print(f"  Figure size: {output_path.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"✗ Error generating visualization: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 6. STATISTICS AND SUMMARY
# ============================================================================

def print_earnings_statistics(df):
    """
    Print comprehensive statistics about earnings features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with earnings features
    """
    print("\n" + "="*80)
    print("[6/6] EARNINGS FEATURE STATISTICS")
    print("="*80)
    
    try:
        # Earnings days analysis
        earnings_days = df[df['is_earnings_day'] == 1]
        
        print(f"\nEarnings Days Summary:")
        print(f"  Total earnings days in dataset: {len(earnings_days)}")
        print(f"  Percentage of trading days: {len(earnings_days)/len(df)*100:.2f}%")
        
        if len(earnings_days) > 0:
            print(f"  Date range: {earnings_days['date'].min().date()} to "
                  f"{earnings_days['date'].max().date()}")
        
        # Days to/since earnings statistics
        if df['days_to_earnings'].notna().sum() > 0:
            print(f"\nDays to Next Earnings:")
            print(f"  Mean: {df['days_to_earnings'].mean():.1f} days")
            print(f"  Median: {df['days_to_earnings'].median():.1f} days")
            print(f"  Min: {df['days_to_earnings'].min():.0f} days")
            print(f"  Max: {df['days_to_earnings'].max():.0f} days (capped at 90)")
        
        if df['days_since_earnings'].notna().sum() > 0:
            print(f"\nDays Since Last Earnings:")
            print(f"  Mean: {df['days_since_earnings'].mean():.1f} days")
            print(f"  Median: {df['days_since_earnings'].median():.1f} days")
            print(f"  Min: {df['days_since_earnings'].min():.0f} days")
            print(f"  Max: {df['days_since_earnings'].max():.0f} days (capped at 90)")
        
        # Earnings surprise statistics
        if df['earnings_surprise'].notna().sum() > 0:
            surprise_data = df['earnings_surprise'].dropna()
            print(f"\nEarnings Surprise (EPS):")
            print(f"  Count: {len(surprise_data)}")
            print(f"  Mean: {surprise_data.mean():.4f}")
            print(f"  Median: {surprise_data.median():.4f}")
            print(f"  Std Dev: {surprise_data.std():.4f}")
            print(f"  Min: {surprise_data.min():.4f}")
            print(f"  Max: {surprise_data.max():.4f}")
            
            positive_surprises = (surprise_data > 0).sum()
            negative_surprises = (surprise_data < 0).sum()
            print(f"  Positive surprises: {positive_surprises} "
                  f"({positive_surprises/len(surprise_data)*100:.1f}%)")
            print(f"  Negative surprises: {negative_surprises} "
                  f"({negative_surprises/len(surprise_data)*100:.1f}%)")
        
        if df['earnings_surprise_pct'].notna().sum() > 0:
            surprise_pct_data = df['earnings_surprise_pct'].dropna()
            print(f"\nEarnings Surprise Percentage:")
            print(f"  Count: {len(surprise_pct_data)}")
            print(f"  Mean: {surprise_pct_data.mean():.2f}%")
            print(f"  Median: {surprise_pct_data.median():.2f}%")
            print(f"  Std Dev: {surprise_pct_data.std():.2f}%")
            print(f"  Min: {surprise_pct_data.min():.2f}%")
            print(f"  Max: {surprise_pct_data.max():.2f}%")
        
        # Price impact analysis on earnings days
        if len(earnings_days) > 0 and 'daily_return' in df.columns:
            earnings_returns = earnings_days['daily_return'].dropna()
            non_earnings_returns = df[df['is_earnings_day'] == 0]['daily_return'].dropna()
            
            if len(earnings_returns) > 0 and len(non_earnings_returns) > 0:
                print(f"\nPrice Impact Analysis:")
                print(f"  Average return on earnings days: {earnings_returns.mean():.4f}%")
                print(f"  Average return on non-earnings days: {non_earnings_returns.mean():.4f}%")
                print(f"  Volatility on earnings days (std): {earnings_returns.std():.4f}%")
                print(f"  Volatility on non-earnings days (std): {non_earnings_returns.std():.4f}%")
                
                # Test if volatility is higher on earnings days
                volatility_ratio = earnings_returns.std() / non_earnings_returns.std()
                print(f"  Volatility ratio (earnings/non-earnings): {volatility_ratio:.2f}x")
        
    except Exception as e:
        print(f"✗ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline for earnings events integration.
    """
    print("\n" + "="*80)
    print("EARNINGS EVENTS INTEGRATION PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'processed'
    output_dir = base_dir / 'reports' / 'figures'
    
    input_csv = data_dir / 'apple_feature_plus_time_lag.csv'
    output_csv = data_dir / 'apple_with_earnings.csv'
    output_fig = output_dir / 'earnings_impact.png'
    
    # Step 1: Load dataset
    df = load_dataset(input_csv)
    if df is None:
        print("\n✗ Failed to load dataset. Exiting.")
        return
    
    # Step 2: Fetch earnings data
    df_earnings = fetch_earnings_data('AAPL')
    if df_earnings is None:
        print("\n⚠ No earnings data available. Creating placeholder features...")
        # Create placeholder features
        df['is_earnings_day'] = 0
        df['days_to_earnings'] = np.nan
        df['days_since_earnings'] = np.nan
        df['earnings_surprise'] = np.nan
        df['earnings_surprise_pct'] = np.nan
        df['revenue_surprise'] = np.nan
    else:
        # Step 3: Engineer earnings features
        df = engineer_earnings_features(df, df_earnings)
        if df is None:
            print("\n✗ Failed to engineer features. Exiting.")
            return
    
    # Step 4: Save dataset
    save_dataset(df, output_csv)
    
    # Step 5: Generate visualization
    if df_earnings is not None:
        plot_earnings_impact(df, output_fig)
    else:
        print("\n⚠ Skipping visualization (no earnings data)")
    
    # Step 6: Print statistics
    if df_earnings is not None:
        print_earnings_statistics(df)
    
    # Final summary
    print("\n" + "="*80)
    print("EARNINGS EVENTS INTEGRATION COMPLETE")
    print("="*80)
    
    print("\n✓ Output Files:")
    print(f"  • {output_csv}")
    if df_earnings is not None:
        print(f"  • {output_fig}")
    
    print("\n✓ Dataset Summary:")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    if df_earnings is not None:
        print(f"\n✓ Earnings Features:")
        print(f"  Earnings days: {df['is_earnings_day'].sum()}")
        print(f"  Surprise data points: {df['earnings_surprise'].notna().sum()}")
    else:
        print(f"\n⚠ Earnings Features: Placeholder values only (all NaN)")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
