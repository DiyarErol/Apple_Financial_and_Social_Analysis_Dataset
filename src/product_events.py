"""
Product Events Integration Script
Apple Financial and Social Analysis Dataset

This script integrates Apple product launch events, WWDC conferences, and special events
into the dataset, computing event proximity features and hype scores.

Features added:
- is_product_launch: Binary flag for product launch dates
- days_to_product_launch: Signed distance to nearest launch (±30 days)
- product_type: Product category (iPhone/iPad/Mac/Watch/Other)
- is_wwdc: Binary flag for WWDC conference dates
- is_apple_event: Binary flag for special Apple events
- event_hype_score: Normalized 7-day trend change around events [-1, +1]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_dataset(csv_path):
    """
    Load the dataset with earnings features.
    
    Parameters:
    -----------
    csv_path : Path or str
        Path to apple_with_earnings.csv
    
    Returns:
    --------
    pd.DataFrame
        Dataset with date normalized
    """
    print("\n" + "="*80)
    print("[1/7] LOADING DATASET")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    print(f"✓ Dataset loaded: {len(df):,} records, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


# ============================================================================
# 2. PRODUCT LAUNCH CALENDAR
# ============================================================================

def get_product_launches():
    """
    Define Apple product launch dates with product names and types.
    
    Returns:
    --------
    pd.DataFrame
        Product launch calendar with date, product_name, product_type
    """
    print("\n" + "="*80)
    print("[2/7] DEFINING PRODUCT LAUNCH CALENDAR")
    print("="*80)
    
    # Comprehensive product launch dictionary
    product_launches = {
        # iPhone Releases
        '2015-09-09': ('iPhone 6S', 'iPhone'),
        '2016-09-07': ('iPhone 7', 'iPhone'),
        '2017-09-12': ('iPhone 8/X', 'iPhone'),
        '2018-09-12': ('iPhone XS', 'iPhone'),
        '2019-09-10': ('iPhone 11', 'iPhone'),
        '2020-10-13': ('iPhone 12', 'iPhone'),
        '2021-09-14': ('iPhone 13', 'iPhone'),
        '2022-09-07': ('iPhone 14', 'iPhone'),
        '2023-09-12': ('iPhone 15', 'iPhone'),
        '2024-09-09': ('iPhone 16', 'iPhone'),
        
        # iPad Releases (major ones)
        '2015-11-11': ('iPad Pro 12.9"', 'iPad'),
        '2017-06-05': ('iPad Pro 10.5"', 'iPad'),
        '2018-11-07': ('iPad Pro (3rd gen)', 'iPad'),
        '2020-03-25': ('iPad Pro (4th gen)', 'iPad'),
        '2021-05-21': ('iPad Pro M1', 'iPad'),
        '2022-10-26': ('iPad Pro M2', 'iPad'),
        
        # Mac Releases (major ones)
        '2016-10-27': ('MacBook Pro Touch Bar', 'Mac'),
        '2018-11-07': ('MacBook Air Retina', 'Mac'),
        '2020-11-17': ('MacBook M1', 'Mac'),
        '2021-10-26': ('MacBook Pro M1 Pro/Max', 'Mac'),
        '2022-06-06': ('MacBook Air M2', 'Mac'),
        '2023-06-05': ('Mac Studio', 'Mac'),
        '2024-03-08': ('MacBook Air M3', 'Mac'),
        
        # Apple Watch Releases
        '2015-04-24': ('Apple Watch', 'Watch'),
        '2016-09-16': ('Apple Watch Series 2', 'Watch'),
        '2017-09-22': ('Apple Watch Series 3', 'Watch'),
        '2018-09-21': ('Apple Watch Series 4', 'Watch'),
        '2019-09-20': ('Apple Watch Series 5', 'Watch'),
        '2020-09-18': ('Apple Watch Series 6', 'Watch'),
        '2021-09-24': ('Apple Watch Series 7', 'Watch'),
        '2022-09-16': ('Apple Watch Series 8', 'Watch'),
        '2023-09-22': ('Apple Watch Series 9', 'Watch'),
        '2024-09-20': ('Apple Watch Series 10', 'Watch'),
    }
    
    # Convert to DataFrame
    launch_data = []
    for date_str, (product_name, product_type) in product_launches.items():
        launch_data.append({
            'date': pd.to_datetime(date_str),
            'product_name': product_name,
            'product_type': product_type
        })
    
    df_launches = pd.DataFrame(launch_data)
    df_launches = df_launches.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Product launch calendar defined: {len(df_launches)} launches")
    print(f"  Date range: {df_launches['date'].min().date()} to {df_launches['date'].max().date()}")
    print(f"\n  Breakdown by product type:")
    for ptype, count in df_launches['product_type'].value_counts().items():
        print(f"    {ptype}: {count} launches")
    
    return df_launches


# ============================================================================
# 3. APPLE EVENTS CALENDAR
# ============================================================================

def get_apple_events():
    """
    Define WWDC and special Apple event dates.
    
    Returns:
    --------
    dict
        Dictionary with 'wwdc' and 'special_events' as lists of datetime objects
    """
    print("\n" + "="*80)
    print("[3/7] DEFINING APPLE EVENTS CALENDAR")
    print("="*80)
    
    # WWDC Dates (typically first week of June)
    wwdc_dates = [
        '2015-06-08',  # WWDC 2015
        '2016-06-13',  # WWDC 2016
        '2017-06-05',  # WWDC 2017
        '2018-06-04',  # WWDC 2018
        '2019-06-03',  # WWDC 2019
        '2020-06-22',  # WWDC 2020 (virtual)
        '2021-06-07',  # WWDC 2021 (virtual)
        '2022-06-06',  # WWDC 2022 (virtual)
        '2023-06-05',  # WWDC 2023
        '2024-06-10',  # WWDC 2024
        '2025-06-09',  # WWDC 2025 (estimated)
    ]
    
    # Special Apple Events (September/October events, Spring events)
    special_events = [
        '2015-09-09',  # iPhone 6S event
        '2016-03-21',  # iPhone SE event
        '2016-09-07',  # iPhone 7 event
        '2017-03-27',  # iPad event
        '2017-09-12',  # iPhone 8/X event
        '2018-03-27',  # iPad event
        '2018-09-12',  # iPhone XS event
        '2018-10-30',  # iPad Pro event
        '2019-03-25',  # Services event
        '2019-09-10',  # iPhone 11 event
        '2020-09-15',  # Apple Watch/iPad event
        '2020-10-13',  # iPhone 12 event
        '2020-11-10',  # M1 Mac event
        '2021-04-20',  # Spring loaded event
        '2021-09-14',  # iPhone 13 event
        '2021-10-18',  # Unleashed event (MacBook Pro)
        '2022-03-08',  # Peek performance event
        '2022-09-07',  # iPhone 14 event
        '2022-10-18',  # iPad Pro event
        '2023-09-12',  # iPhone 15 event
        '2023-10-30',  # Scary fast event (M3)
        '2024-05-07',  # iPad Pro M4 event
        '2024-09-09',  # iPhone 16 event
    ]
    
    # Convert to datetime
    wwdc_dates = [pd.to_datetime(d) for d in wwdc_dates]
    special_events = [pd.to_datetime(d) for d in special_events]
    
    print(f"✓ WWDC dates defined: {len(wwdc_dates)} conferences")
    print(f"✓ Special events defined: {len(special_events)} events")
    
    return {
        'wwdc': wwdc_dates,
        'special_events': special_events
    }


# ============================================================================
# 4. FEATURE ENGINEERING - PRODUCT EVENTS
# ============================================================================

def engineer_product_features(df, df_launches, events_dict):
    """
    Add product event features to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Main dataset
    df_launches : pd.DataFrame
        Product launch calendar
    events_dict : dict
        Dictionary with WWDC and special event dates
    
    Returns:
    --------
    pd.DataFrame
        Dataset with new event features
    """
    print("\n" + "="*80)
    print("[4/7] ENGINEERING PRODUCT EVENT FEATURES")
    print("="*80)
    
    df = df.copy()
    
    # Extract launch dates
    launch_dates = df_launches['date'].values
    wwdc_dates = events_dict['wwdc']
    special_events = events_dict['special_events']
    
    print(f"\nProcessing {len(launch_dates)} product launches...")
    
    # 1. is_product_launch
    print("  Computing is_product_launch...")
    df['is_product_launch'] = df['date'].isin(launch_dates).astype(int)
    launch_count = df['is_product_launch'].sum()
    print(f"    ✓ Found {launch_count} product launch days in dataset")
    
    # 2. days_to_product_launch (signed distance to nearest launch within ±30 days)
    print("  Computing days_to_product_launch...")
    days_to_launch = []
    
    for date in df['date']:
        # Calculate days to all launches
        days_diff = [(pd.Timestamp(ld) - pd.Timestamp(date)).days for ld in launch_dates]
        
        # Filter to ±30 days window
        nearby_launches = [d for d in days_diff if abs(d) <= 30]
        
        if nearby_launches:
            # Get signed distance to nearest launch
            nearest = min(nearby_launches, key=abs)
            days_to_launch.append(nearest)
        else:
            days_to_launch.append(np.nan)
    
    df['days_to_product_launch'] = days_to_launch
    non_null_count = df['days_to_product_launch'].notna().sum()
    print(f"    ✓ days_to_product_launch: {non_null_count:,} non-null values (within ±30 day window)")
    
    # 3. product_type (merge from launch calendar)
    print("  Computing product_type...")
    # Create mapping from date to product type
    type_map = dict(zip(df_launches['date'], df_launches['product_type']))
    df['product_type'] = df['date'].map(type_map).fillna('None')
    
    type_counts = df[df['product_type'] != 'None']['product_type'].value_counts()
    print(f"    ✓ product_type assigned:")
    for ptype, count in type_counts.items():
        print(f"      {ptype}: {count} days")
    
    # 4. is_wwdc
    print("  Computing is_wwdc...")
    df['is_wwdc'] = df['date'].isin(wwdc_dates).astype(int)
    wwdc_count = df['is_wwdc'].sum()
    print(f"    ✓ Found {wwdc_count} WWDC days in dataset")
    
    # 5. is_apple_event
    print("  Computing is_apple_event...")
    df['is_apple_event'] = df['date'].isin(special_events).astype(int)
    event_count = df['is_apple_event'].sum()
    print(f"    ✓ Found {event_count} special event days in dataset")
    
    print(f"\n✓ Product event features added:")
    print(f"  - is_product_launch: {launch_count} launch days")
    print(f"  - days_to_product_launch: within ±30 day window")
    print(f"  - product_type: {len(type_counts)} product categories")
    print(f"  - is_wwdc: {wwdc_count} WWDC days")
    print(f"  - is_apple_event: {event_count} special event days")
    
    return df


# ============================================================================
# 5. FEATURE ENGINEERING - EVENT HYPE SCORE
# ============================================================================

def compute_event_hype_score(df):
    """
    Compute event hype score based on Google Trends data.
    
    The hype score measures the 7-day percentage change in search interest
    (apple iphone + macbook) around each date, normalized and clipped to [-1, +1].
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with Google Trends columns
    
    Returns:
    --------
    pd.DataFrame
        Dataset with event_hype_score column
    """
    print("\n" + "="*80)
    print("[5/7] COMPUTING EVENT HYPE SCORE")
    print("="*80)
    
    df = df.copy()
    
    # Check if Google Trends columns exist
    trend_cols = ['apple iphone', 'macbook']
    missing_cols = [col for col in trend_cols if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠ Missing Google Trends columns: {missing_cols}")
        print(f"  Creating placeholder event_hype_score with zeros")
        df['event_hype_score'] = 0.0
        return df
    
    print("  Computing combined trend score (apple iphone + macbook)...")
    
    # Combine trend scores (fill NaN with 0)
    df['combined_trend'] = (
        df['apple iphone'].fillna(0) + 
        df['macbook'].fillna(0)
    )
    
    # Compute 7-day percentage change
    print("  Computing 7-day percentage change...")
    df['trend_pct_change_7d'] = df['combined_trend'].pct_change(periods=7) * 100
    
    # Normalize to [-1, +1] using robust scaling (clip extreme values first)
    print("  Normalizing and clipping to [-1, +1]...")
    pct_change = df['trend_pct_change_7d'].fillna(0)
    
    # Clip extreme outliers (beyond 3 standard deviations)
    mean_change = pct_change.mean()
    std_change = pct_change.std()
    lower_bound = mean_change - 3 * std_change
    upper_bound = mean_change + 3 * std_change
    
    pct_change_clipped = pct_change.clip(lower_bound, upper_bound)
    
    # Normalize to [-1, +1]
    if pct_change_clipped.max() > pct_change_clipped.min():
        normalized = 2 * (pct_change_clipped - pct_change_clipped.min()) / \
                     (pct_change_clipped.max() - pct_change_clipped.min()) - 1
    else:
        normalized = pct_change_clipped * 0  # All zeros if no variation
    
    df['event_hype_score'] = normalized.clip(-1, 1)
    
    # Clean up temporary columns
    df = df.drop(columns=['combined_trend', 'trend_pct_change_7d'])
    
    # Statistics
    hype_stats = df['event_hype_score'].describe()
    print(f"\n✓ Event hype score computed:")
    print(f"  Mean: {hype_stats['mean']:.3f}")
    print(f"  Std: {hype_stats['std']:.3f}")
    print(f"  Min: {hype_stats['min']:.3f}")
    print(f"  Max: {hype_stats['max']:.3f}")
    print(f"  Range: [{-1:.1f}, {1:.1f}] (clipped)")
    
    return df


# ============================================================================
# 6. DATA SAVING
# ============================================================================

def save_dataset(df, output_path):
    """
    Save the dataset with product event features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with event features
    output_path : Path or str
        Output CSV path
    """
    print("\n" + "="*80)
    print("[6/7] SAVING DATASET")
    print("="*80)
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    file_size_kb = output_path.stat().st_size / 1024
    
    print(f"✓ Dataset saved: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  File size: {file_size_kb:.1f} KB")
    
    # List new columns
    new_cols = ['is_product_launch', 'days_to_product_launch', 'product_type',
                'is_wwdc', 'is_apple_event', 'event_hype_score']
    print(f"\n  New columns added: {new_cols}")


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def plot_event_hype_overlay(df, df_launches, output_path):
    """
    Create visualization of stock price with event hype score and product launches.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with event features
    df_launches : pd.DataFrame
        Product launch calendar
    output_path : Path or str
        Output figure path
    """
    print("\n" + "="*80)
    print("[7/7] GENERATING VISUALIZATION")
    print("="*80)
    
    print("\nPlotting event hype overlay...")
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot 1: Stock price
    color1 = '#1f77b4'
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Close Price ($)', fontsize=12, fontweight='bold', color=color1)
    ax1.plot(df['date'], df['close'], color=color1, linewidth=1.5, label='Close Price', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Event hype score
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Event Hype Score', fontsize=12, fontweight='bold', color=color2)
    ax2.plot(df['date'], df['event_hype_score'], color=color2, linewidth=1.0, 
             label='Event Hype Score', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Mark product launch dates with vertical lines
    launch_dates = df_launches['date']
    product_types = df_launches['product_type']
    
    # Color mapping for product types
    type_colors = {
        'iPhone': '#e74c3c',  # Red
        'iPad': '#3498db',    # Blue
        'Mac': '#9b59b6',     # Purple
        'Watch': '#2ecc71',   # Green
    }
    
    print(f"  Marking {len(launch_dates)} product launch dates...")
    
    for launch_date, ptype in zip(launch_dates, product_types):
        if df['date'].min() <= launch_date <= df['date'].max():
            color = type_colors.get(ptype, '#95a5a6')
            ax1.axvline(x=launch_date, color=color, linestyle='-', linewidth=1.5, 
                       alpha=0.7, label=ptype if ptype not in ax1.get_legend_handles_labels()[1] else '')
    
    # Title and legends
    plt.title('Apple Stock Price with Product Launch Events and Hype Score\n' +
              'Vertical lines indicate product launches (iPhone/iPad/Mac/Watch)',
              fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Remove duplicate product type labels
    seen = set()
    unique_lines = []
    unique_labels = []
    for line, label in zip(lines1 + lines2, labels1 + labels2):
        if label not in seen:
            seen.add(label)
            unique_lines.append(line)
            unique_labels.append(label)
    
    ax1.legend(unique_lines, unique_labels, loc='upper left', fontsize=10, framealpha=0.9)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    file_size_kb = output_path.stat().st_size / 1024
    
    print(f"\n✓ Visualization saved: {output_path}")
    print(f"  Figure size: {file_size_kb:.1f} KB")
    print(f"  Resolution: 300 DPI")


# ============================================================================
# 8. STATISTICS
# ============================================================================

def print_event_statistics(df, df_launches):
    """
    Print comprehensive statistics about product events.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with event features
    df_launches : pd.DataFrame
        Product launch calendar
    """
    print("\n" + "="*80)
    print("PRODUCT EVENT FEATURE STATISTICS")
    print("="*80)
    
    # Product launch summary
    print("\nProduct Launch Summary:")
    print(f"  Total product launches in dataset: {df['is_product_launch'].sum()}")
    print(f"  Percentage of trading days: {df['is_product_launch'].mean() * 100:.2f}%")
    
    # Product type breakdown
    print("\nProduct Type Breakdown:")
    type_counts = df[df['product_type'] != 'None']['product_type'].value_counts()
    for ptype, count in type_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {ptype}: {count} days ({pct:.3f}%)")
    
    # Days to product launch
    days_to_launch = df['days_to_product_launch'].dropna()
    if len(days_to_launch) > 0:
        print("\nDays to Product Launch (±30 day window):")
        print(f"  Non-null values: {len(days_to_launch):,} ({len(days_to_launch)/len(df)*100:.1f}%)")
        print(f"  Mean: {days_to_launch.mean():.1f} days")
        print(f"  Median: {days_to_launch.median():.1f} days")
        print(f"  Range: [{days_to_launch.min():.0f}, {days_to_launch.max():.0f}] days")
    
    # Apple events
    print("\nApple Events Summary:")
    print(f"  WWDC days: {df['is_wwdc'].sum()}")
    print(f"  Special event days: {df['is_apple_event'].sum()}")
    print(f"  Total event days: {(df['is_wwdc'] | df['is_apple_event']).sum()}")
    
    # Event hype score
    hype_score = df['event_hype_score']
    print("\nEvent Hype Score Statistics:")
    print(f"  Mean: {hype_score.mean():.3f}")
    print(f"  Median: {hype_score.median():.3f}")
    print(f"  Std: {hype_score.std():.3f}")
    print(f"  Range: [{hype_score.min():.3f}, {hype_score.max():.3f}]")
    
    # High hype periods (>0.5)
    high_hype = df[hype_score > 0.5]
    print(f"\nHigh Hype Periods (score > 0.5):")
    print(f"  Count: {len(high_hype)} days ({len(high_hype)/len(df)*100:.1f}%)")
    if len(high_hype) > 0:
        print(f"  Average close price: ${high_hype['close'].mean():.2f}")
        print(f"  Average volume: {high_hype['volume'].mean():,.0f}")


# ============================================================================
# 9. MAIN
# ============================================================================

def main():
    """Main execution pipeline."""
    
    print("\n" + "="*80)
    print("PRODUCT EVENTS INTEGRATION PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("="*80)
    
    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data' / 'processed'
    reports_dir = script_dir / 'reports' / 'figures'
    
    input_csv = data_dir / 'apple_with_earnings.csv'
    output_csv = data_dir / 'apple_with_events.csv'
    output_fig = reports_dir / 'event_hype_overlay.png'
    
    # 1. Load dataset
    df = load_dataset(input_csv)
    
    # 2. Get product launch calendar
    df_launches = get_product_launches()
    
    # 3. Get Apple events calendar
    events_dict = get_apple_events()
    
    # 4. Engineer product event features
    df = engineer_product_features(df, df_launches, events_dict)
    
    # 5. Compute event hype score
    df = compute_event_hype_score(df)
    
    # 6. Save dataset
    save_dataset(df, output_csv)
    
    # 7. Generate visualization
    plot_event_hype_overlay(df, df_launches, output_fig)
    
    # 8. Print statistics
    print_event_statistics(df, df_launches)
    
    # Final summary
    print("\n" + "="*80)
    print("PRODUCT EVENTS INTEGRATION COMPLETE")
    print("="*80)
    
    print(f"\n✓ Output Files:")
    print(f"  • {output_csv}")
    print(f"  • {output_fig}")
    
    print(f"\n✓ Dataset Summary:")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    print(f"\n✓ Product Event Features:")
    print(f"  Product launches: {df['is_product_launch'].sum()}")
    print(f"  WWDC conferences: {df['is_wwdc'].sum()}")
    print(f"  Special events: {df['is_apple_event'].sum()}")
    print(f"  Event hype score: normalized trend change [-1, +1]")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
