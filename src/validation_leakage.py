"""
Data Leakage Validation Script
Apple Financial and Social Analysis Dataset

This script performs comprehensive checks to detect potential data leakage in the dataset,
ensuring that all features use only past information and targets are properly shifted.

Validation checks:
1. Rolling/lag features use only past data (right-aligned windows)
2. Target variables are strictly future-shifted with no accidental merges
3. Feature-target correlations at negative lags (detecting future information)
4. Technical indicators (VWAP, OBV) computed from past data only
5. Temporal consistency of date-based features
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_dataset(csv_path):
    """
    Load the dataset for validation.
    
    Parameters:
    -----------
    csv_path : Path or str
        Path to apple_with_events.csv
    
    Returns:
    --------
    pd.DataFrame
        Dataset with date normalized
    """
    print("\n" + "="*80)
    print("DATA LEAKAGE VALIDATION PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("="*80)
    
    print("\n[1/6] LOADING DATASET")
    print("-" * 80)
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Dataset loaded: {len(df):,} records, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Columns: {len(df.columns)}")
    
    return df


# ============================================================================
# 2. LEAKAGE CHECK: LAG AND ROLLING FEATURES
# ============================================================================

def check_lag_rolling_features(df):
    """
    Validate that lag and rolling features only use past data.
    
    Checks:
    - Lag features are properly shifted (no current period data)
    - Rolling windows are right-aligned (use only past data)
    - No forward-looking information in window computations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
    
    Returns:
    --------
    dict
        Validation results with status and details
    """
    print("\n[2/6] VALIDATING LAG AND ROLLING FEATURES")
    print("-" * 80)
    
    results = {
        'status': 'PASS',
        'checks': [],
        'warnings': [],
        'failures': []
    }
    
    # Identify lag and rolling columns
    lag_cols = [col for col in df.columns if 'lag' in col.lower()]
    rolling_cols = [col for col in df.columns if 'rolling' in col.lower()]
    
    print(f"\nFound {len(lag_cols)} lag features and {len(rolling_cols)} rolling features")
    
    # Check 1: Lag features correlation with source
    print("\n  Check 2.1: Lag feature temporal consistency")
    for lag_col in lag_cols:
        # Extract source column and lag period
        if 'close_lag_' in lag_col:
            source_col = 'close'
            lag_period = int(lag_col.split('_')[-1])
        elif 'volume_lag_' in lag_col:
            source_col = 'volume'
            lag_period = int(lag_col.split('_')[-1])
        elif 'return_lag_' in lag_col:
            source_col = 'daily_return'
            lag_period = int(lag_col.split('_')[-1])
        else:
            continue
        
        # Verify lag alignment: lag_col[t] should equal source_col[t-lag_period]
        if source_col in df.columns:
            # Shift source column and compare
            expected = df[source_col].shift(lag_period)
            actual = df[lag_col]
            
            # Compare (allowing for NaN in first lag_period rows)
            valid_idx = ~(expected.isna() | actual.isna())
            if valid_idx.sum() > 0:
                correlation = expected[valid_idx].corr(actual[valid_idx])
                
                if correlation > 0.9999:  # Should be nearly perfect
                    results['checks'].append(f"    ✓ {lag_col}: Properly shifted (corr={correlation:.6f})")
                else:
                    msg = f"    ✗ {lag_col}: Suspicious alignment (corr={correlation:.6f}, expected ~1.0)"
                    results['failures'].append(msg)
                    results['status'] = 'FAIL'
                    print(msg)
    
    if not results['failures']:
        print("    ✓ All lag features properly aligned")
    
    # Check 2: Rolling features should not have future correlation
    print("\n  Check 2.2: Rolling feature temporal validity")
    base_cols = ['close', 'volume', 'daily_return']
    
    for rolling_col in rolling_cols:
        # Check if rolling feature at time t correlates with future base values
        for base_col in base_cols:
            if base_col not in df.columns:
                continue
            
            # Test: rolling_col[t] should NOT correlate with base_col[t+k] for k>0
            future_shifts = [1, 2, 5]
            for shift in future_shifts:
                future_base = df[base_col].shift(-shift)
                valid_idx = ~(future_base.isna() | df[rolling_col].isna())
                
                if valid_idx.sum() > 100:  # Need sufficient data
                    corr = df[rolling_col][valid_idx].corr(future_base[valid_idx])
                    
                    # High correlation with future is suspicious
                    if abs(corr) > 0.7:
                        msg = f"    ⚠ {rolling_col} corr with future {base_col}[t+{shift}]: {corr:.3f}"
                        results['warnings'].append(msg)
                        print(msg)
    
    if not results['warnings']:
        print("    ✓ No suspicious future correlations in rolling features")
    
    # Check 3: Rolling windows should increase NaN count at beginning
    print("\n  Check 2.3: Rolling window NaN pattern")
    for rolling_col in rolling_cols:
        # Extract window size from column name
        if 'rolling_max_30d' in rolling_col or 'rolling_min_30d' in rolling_col:
            expected_nan_start = 29  # 30-day window needs 29 previous days
        else:
            expected_nan_start = 0
        
        # Count leading NaNs
        first_valid_idx = df[rolling_col].first_valid_index()
        if first_valid_idx is not None:
            actual_nan_start = first_valid_idx
            
            if actual_nan_start >= expected_nan_start:
                results['checks'].append(f"    ✓ {rolling_col}: {actual_nan_start} leading NaNs (expected ~{expected_nan_start})")
            else:
                msg = f"    ⚠ {rolling_col}: Only {actual_nan_start} leading NaNs (expected ~{expected_nan_start})"
                results['warnings'].append(msg)
                print(msg)
    
    if len(results['checks']) > 0 and not results['failures']:
        print("    ✓ Rolling window NaN patterns consistent with right-alignment")
    
    return results


# ============================================================================
# 3. LEAKAGE CHECK: TARGET VARIABLES
# ============================================================================

def check_target_variables(df):
    """
    Validate that target variables are strictly future-shifted.
    
    Checks:
    - next_day_* targets are shifted -1 from source
    - next_week_* targets are shifted -5 from source
    - direction target matches sign of next_day_return
    - No accidental current-period information
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
    
    Returns:
    --------
    dict
        Validation results
    """
    print("\n[3/6] VALIDATING TARGET VARIABLES")
    print("-" * 80)
    
    results = {
        'status': 'PASS',
        'checks': [],
        'warnings': [],
        'failures': []
    }
    
    # Identify target columns
    target_cols = [col for col in df.columns if col.startswith('next_') or col == 'direction']
    print(f"\nFound {len(target_cols)} target variables: {target_cols}")
    
    # Check 1: next_day_close should be close shifted -1
    print("\n  Check 3.1: next_day_close temporal alignment")
    if 'next_day_close' in df.columns and 'close' in df.columns:
        expected = df['close'].shift(-1)
        actual = df['next_day_close']
        
        valid_idx = ~(expected.isna() | actual.isna())
        if valid_idx.sum() > 0:
            correlation = expected[valid_idx].corr(actual[valid_idx])
            max_diff = (expected[valid_idx] - actual[valid_idx]).abs().max()
            
            if correlation > 0.9999 and max_diff < 0.01:
                results['checks'].append(f"    ✓ next_day_close: Properly shifted (corr={correlation:.6f}, max_diff=${max_diff:.4f})")
                print(f"    ✓ next_day_close: Properly shifted (corr={correlation:.6f})")
            else:
                msg = f"    ✗ next_day_close: Suspicious alignment (corr={correlation:.6f})"
                results['failures'].append(msg)
                results['status'] = 'FAIL'
                print(msg)
    
    # Check 2: next_day_return should be computed from next_day_close
    print("\n  Check 3.2: next_day_return consistency")
    if all(col in df.columns for col in ['next_day_return', 'close', 'next_day_close']):
        # Verify: next_day_return = (next_day_close - close) / close * 100
        expected_return = ((df['next_day_close'] - df['close']) / df['close']) * 100
        actual_return = df['next_day_return']
        
        valid_idx = ~(expected_return.isna() | actual_return.isna())
        if valid_idx.sum() > 0:
            max_diff = (expected_return[valid_idx] - actual_return[valid_idx]).abs().max()
            
            if max_diff < 0.01:
                results['checks'].append(f"    ✓ next_day_return: Consistent with close prices (max_diff={max_diff:.6f}%)")
                print(f"    ✓ next_day_return: Consistent (max_diff={max_diff:.6f}%)")
            else:
                msg = f"    ⚠ next_day_return: Inconsistency detected (max_diff={max_diff:.4f}%)"
                results['warnings'].append(msg)
                print(msg)
    
    # Check 3: next_week_close should be close shifted -5
    print("\n  Check 3.3: next_week_close temporal alignment")
    if 'next_week_close' in df.columns and 'close' in df.columns:
        expected = df['close'].shift(-5)
        actual = df['next_week_close']
        
        valid_idx = ~(expected.isna() | actual.isna())
        if valid_idx.sum() > 0:
            correlation = expected[valid_idx].corr(actual[valid_idx])
            
            if correlation > 0.99:
                results['checks'].append(f"    ✓ next_week_close: Properly shifted (corr={correlation:.6f})")
                print(f"    ✓ next_week_close: Properly shifted (corr={correlation:.6f})")
            else:
                msg = f"    ⚠ next_week_close: Lower correlation (corr={correlation:.6f})"
                results['warnings'].append(msg)
                print(msg)
    
    # Check 4: direction should match sign of next_day_return
    print("\n  Check 3.4: direction classification consistency")
    if 'direction' in df.columns and 'next_day_return' in df.columns:
        # direction should be 1 when next_day_return > 0, else 0
        expected_direction = (df['next_day_return'] > 0).astype(int)
        actual_direction = df['direction']
        
        valid_idx = ~(expected_direction.isna() | actual_direction.isna())
        if valid_idx.sum() > 0:
            match_rate = (expected_direction[valid_idx] == actual_direction[valid_idx]).mean()
            
            if match_rate > 0.999:
                results['checks'].append(f"    ✓ direction: Consistent with next_day_return ({match_rate*100:.2f}% match)")
                print(f"    ✓ direction: Consistent ({match_rate*100:.2f}% match)")
            else:
                msg = f"    ✗ direction: Inconsistent with next_day_return ({match_rate*100:.2f}% match)"
                results['failures'].append(msg)
                results['status'] = 'FAIL'
                print(msg)
    
    # Check 5: Targets should have NaN at the end (future unknowns)
    print("\n  Check 3.5: Target NaN pattern (future unknowns)")
    for target_col in target_cols:
        if target_col == 'direction':
            continue
        
        last_valid_idx = df[target_col].last_valid_index()
        last_row_idx = df.index[-1]
        
        if last_valid_idx is not None and last_valid_idx < last_row_idx:
            trailing_nans = last_row_idx - last_valid_idx
            results['checks'].append(f"    ✓ {target_col}: {trailing_nans} trailing NaNs (expected for future periods)")
            print(f"    ✓ {target_col}: {trailing_nans} trailing NaNs")
        else:
            msg = f"    ⚠ {target_col}: No trailing NaNs (suspicious - all future values known?)"
            results['warnings'].append(msg)
            print(msg)
    
    return results


# ============================================================================
# 4. LEAKAGE CHECK: FUTURE CORRELATION ANALYSIS
# ============================================================================

def check_future_correlations(df):
    """
    Check for suspicious correlations between features and future targets.
    
    High correlation at negative lags indicates potential leakage where
    features contain future information about targets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
    
    Returns:
    --------
    dict
        Validation results
    """
    print("\n[4/6] CHECKING FUTURE CORRELATIONS")
    print("-" * 80)
    
    results = {
        'status': 'PASS',
        'checks': [],
        'warnings': [],
        'failures': []
    }
    
    # Identify feature and target columns
    target_cols = ['next_day_close', 'next_day_return', 'next_week_close']
    feature_cols = [col for col in df.columns if col not in target_cols + ['date', 'direction']]
    
    # Remove columns that are inherently future-looking
    exclude_patterns = ['next_', 'future_', 'forward_']
    feature_cols = [col for col in feature_cols if not any(pat in col.lower() for pat in exclude_patterns)]
    
    print(f"\nAnalyzing {len(feature_cols)} features against {len(target_cols)} targets")
    print("  Testing lags: +1, +2, +5 days (feature at t vs target at t-lag)")
    
    suspicious_pairs = []
    
    # Test each feature-target pair at future lags
    for target_col in target_cols:
        if target_col not in df.columns:
            continue
        
        print(f"\n  Testing correlations with {target_col}:")
        
        for lag in [1, 2, 5]:
            high_corr_features = []
            
            for feature_col in feature_cols:
                if feature_col not in df.columns:
                    continue
                
                # Shift target backwards (feature[t] vs target[t-lag])
                # If correlation is high, feature might contain future info
                shifted_target = df[target_col].shift(lag)
                valid_idx = ~(df[feature_col].isna() | shifted_target.isna())
                
                if valid_idx.sum() > 100:
                    corr = df[feature_col][valid_idx].corr(shifted_target[valid_idx])
                    
                    # Flag high correlations (>0.8 is very suspicious)
                    if abs(corr) > 0.8:
                        high_corr_features.append((feature_col, corr))
                        suspicious_pairs.append({
                            'feature': feature_col,
                            'target': target_col,
                            'lag': lag,
                            'correlation': corr
                        })
            
            if high_corr_features:
                print(f"    ⚠ Lag +{lag}: {len(high_corr_features)} features with |corr| > 0.8")
                for feat, corr in high_corr_features[:3]:  # Show top 3
                    print(f"      • {feat}: {corr:.3f}")
                    msg = f"    ⚠ {feat} -> {target_col} at lag +{lag}: corr={corr:.3f}"
                    results['warnings'].append(msg)
            else:
                results['checks'].append(f"    ✓ Lag +{lag}: No suspicious correlations (|corr| < 0.8)")
    
    if suspicious_pairs:
        print(f"\n  ⚠ Found {len(suspicious_pairs)} suspicious feature-target pairs")
        results['status'] = 'WARNING'
    else:
        print("\n  ✓ No suspicious future correlations detected")
    
    results['suspicious_pairs'] = suspicious_pairs
    
    return results


# ============================================================================
# 5. LEAKAGE CHECK: TECHNICAL INDICATORS
# ============================================================================

def check_technical_indicators(df):
    """
    Validate that technical indicators (VWAP, OBV) use only past data.
    
    Checks:
    - VWAP computed from cumulative past data
    - OBV computed from cumulative past data
    - No forward-looking information in computations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
    
    Returns:
    --------
    dict
        Validation results
    """
    print("\n[5/6] VALIDATING TECHNICAL INDICATORS")
    print("-" * 80)
    
    results = {
        'status': 'PASS',
        'checks': [],
        'warnings': [],
        'failures': []
    }
    
    # Check 1: VWAP should be monotonically related to price/volume
    print("\n  Check 5.1: VWAP temporal consistency")
    if 'vwap_20' in df.columns:
        # VWAP should not have perfect correlation with future prices
        if 'close' in df.columns:
            future_close = df['close'].shift(-5)
            valid_idx = ~(df['vwap_20'].isna() | future_close.isna())
            
            if valid_idx.sum() > 100:
                corr = df['vwap_20'][valid_idx].corr(future_close[valid_idx])
                
                # High correlation with distant future is suspicious
                if abs(corr) > 0.95:
                    msg = f"    ⚠ VWAP has high correlation with close[t+5]: {corr:.3f}"
                    results['warnings'].append(msg)
                    print(msg)
                else:
                    results['checks'].append(f"    ✓ VWAP correlation with close[t+5]: {corr:.3f} (acceptable)")
                    print(f"    ✓ VWAP correlation with close[t+5]: {corr:.3f}")
    
    # Check 2: OBV should be cumulative (monotonic or near-monotonic)
    print("\n  Check 5.2: OBV temporal consistency")
    if 'obv' in df.columns:
        # OBV should not decrease dramatically (it's cumulative)
        obv_values = df['obv'].dropna()
        if len(obv_values) > 100:
            # Check for suspicious resets (large negative jumps)
            obv_diff = obv_values.diff()
            large_decreases = (obv_diff < -obv_values.mean() * 0.5).sum()
            
            if large_decreases > 10:
                msg = f"    ⚠ OBV has {large_decreases} large decreases (suspicious for cumulative indicator)"
                results['warnings'].append(msg)
                print(msg)
            else:
                results['checks'].append(f"    ✓ OBV appears cumulative ({large_decreases} large decreases)")
                print(f"    ✓ OBV appears cumulative")
        
        # OBV should not perfectly predict future prices
        if 'close' in df.columns:
            future_close = df['close'].shift(-5)
            valid_idx = ~(df['obv'].isna() | future_close.isna())
            
            if valid_idx.sum() > 100:
                corr = df['obv'][valid_idx].corr(future_close[valid_idx])
                
                if abs(corr) > 0.9:
                    msg = f"    ⚠ OBV has very high correlation with close[t+5]: {corr:.3f}"
                    results['warnings'].append(msg)
                    print(msg)
                else:
                    results['checks'].append(f"    ✓ OBV correlation with close[t+5]: {corr:.3f} (acceptable)")
                    print(f"    ✓ OBV correlation with close[t+5]: {corr:.3f}")
    
    # Check 3: Technical indicators should not be perfectly predictive
    print("\n  Check 5.3: Technical indicator predictive power")
    tech_indicators = ['rsi_14', 'macd', 'macd_signal', 'atr_14', 'bb_width']
    target_col = 'next_day_return'
    
    if target_col in df.columns:
        for indicator in tech_indicators:
            if indicator not in df.columns:
                continue
            
            valid_idx = ~(df[indicator].isna() | df[target_col].isna())
            if valid_idx.sum() > 100:
                corr = df[indicator][valid_idx].corr(df[target_col][valid_idx])
                
                # Very high correlation with next day return is suspicious
                if abs(corr) > 0.7:
                    msg = f"    ⚠ {indicator} has high correlation with {target_col}: {corr:.3f}"
                    results['warnings'].append(msg)
                    print(msg)
                else:
                    results['checks'].append(f"    ✓ {indicator} correlation with {target_col}: {corr:.3f}")
    
    if not results['warnings']:
        print("    ✓ All technical indicators show acceptable predictive power")
    
    return results


# ============================================================================
# 6. LEAKAGE CHECK: TEMPORAL CONSISTENCY
# ============================================================================

def check_temporal_consistency(df):
    """
    Validate temporal consistency of date-based features.
    
    Checks:
    - Date features match actual dates (day of week, month, etc.)
    - Event flags are temporally consistent
    - No retroactive information in event features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
    
    Returns:
    --------
    dict
        Validation results
    """
    print("\n[6/6] VALIDATING TEMPORAL CONSISTENCY")
    print("-" * 80)
    
    results = {
        'status': 'PASS',
        'checks': [],
        'warnings': [],
        'failures': []
    }
    
    # Check 1: Day of week consistency
    print("\n  Check 6.1: Day of week consistency")
    if 'dow' in df.columns and 'date' in df.columns:
        expected_dow = df['date'].dt.dayofweek
        actual_dow = df['dow']
        
        valid_idx = ~(expected_dow.isna() | actual_dow.isna())
        if valid_idx.sum() > 0:
            match_rate = (expected_dow[valid_idx] == actual_dow[valid_idx]).mean()
            
            if match_rate > 0.999:
                results['checks'].append(f"    ✓ dow: Matches date.dayofweek ({match_rate*100:.2f}%)")
                print(f"    ✓ dow: Consistent ({match_rate*100:.2f}% match)")
            else:
                msg = f"    ✗ dow: Inconsistent with date ({match_rate*100:.2f}% match)"
                results['failures'].append(msg)
                results['status'] = 'FAIL'
                print(msg)
    
    # Check 2: Month consistency
    print("\n  Check 6.2: Month consistency")
    if 'month' in df.columns and 'date' in df.columns:
        expected_month = df['date'].dt.month
        actual_month = df['month']
        
        valid_idx = ~(expected_month.isna() | actual_month.isna())
        if valid_idx.sum() > 0:
            match_rate = (expected_month[valid_idx] == actual_month[valid_idx]).mean()
            
            if match_rate > 0.999:
                results['checks'].append(f"    ✓ month: Matches date.month ({match_rate*100:.2f}%)")
                print(f"    ✓ month: Consistent ({match_rate*100:.2f}% match)")
            else:
                msg = f"    ✗ month: Inconsistent with date ({match_rate*100:.2f}% match)"
                results['failures'].append(msg)
                results['status'] = 'FAIL'
                print(msg)
    
    # Check 3: Quarter consistency
    print("\n  Check 6.3: Quarter consistency")
    if 'quarter' in df.columns and 'date' in df.columns:
        expected_quarter = df['date'].dt.quarter
        actual_quarter = df['quarter']
        
        valid_idx = ~(expected_quarter.isna() | actual_quarter.isna())
        if valid_idx.sum() > 0:
            match_rate = (expected_quarter[valid_idx] == actual_quarter[valid_idx]).mean()
            
            if match_rate > 0.999:
                results['checks'].append(f"    ✓ quarter: Matches date.quarter ({match_rate*100:.2f}%)")
                print(f"    ✓ quarter: Consistent ({match_rate*100:.2f}% match)")
            else:
                msg = f"    ✗ quarter: Inconsistent with date ({match_rate*100:.2f}% match)"
                results['failures'].append(msg)
                results['status'] = 'FAIL'
                print(msg)
    
    # Check 4: Year consistency
    print("\n  Check 6.4: Year consistency")
    if 'year' in df.columns and 'date' in df.columns:
        expected_year = df['date'].dt.year
        actual_year = df['year']
        
        valid_idx = ~(expected_year.isna() | actual_year.isna())
        if valid_idx.sum() > 0:
            match_rate = (expected_year[valid_idx] == actual_year[valid_idx]).mean()
            
            if match_rate > 0.999:
                results['checks'].append(f"    ✓ year: Matches date.year ({match_rate*100:.2f}%)")
                print(f"    ✓ year: Consistent ({match_rate*100:.2f}% match)")
            else:
                msg = f"    ✗ year: Inconsistent with date ({match_rate*100:.2f}% match)"
                results['failures'].append(msg)
                results['status'] = 'FAIL'
                print(msg)
    
    # Check 5: Event flags should not predict future returns too well
    print("\n  Check 6.5: Event flag predictive power")
    event_flags = ['is_product_launch', 'is_wwdc', 'is_apple_event', 'is_earnings_day']
    target_col = 'next_day_return'
    
    if target_col in df.columns:
        for flag in event_flags:
            if flag not in df.columns:
                continue
            
            event_days = df[df[flag] == 1]
            if len(event_days) > 5:
                avg_return = event_days[target_col].mean()
                overall_return = df[target_col].mean()
                
                # Event days having dramatically different returns might indicate leakage
                if abs(avg_return - overall_return) > 5.0:  # >5% difference
                    msg = f"    ⚠ {flag}: Large return difference (event: {avg_return:.2f}%, overall: {overall_return:.2f}%)"
                    results['warnings'].append(msg)
                    print(msg)
                else:
                    results['checks'].append(f"    ✓ {flag}: Reasonable return profile")
    
    if not results['failures'] and not results['warnings']:
        print("\n    ✓ All temporal features are consistent")
    
    return results


# ============================================================================
# 7. REPORT GENERATION
# ============================================================================

def generate_report(all_results, df, output_path):
    """
    Generate comprehensive leakage validation report.
    
    Parameters:
    -----------
    all_results : dict
        Results from all validation checks
    df : pd.DataFrame
        Dataset being validated
    output_path : Path or str
        Output report path
    """
    print("\n" + "="*80)
    print("GENERATING VALIDATION REPORT")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("DATA LEAKAGE VALIDATION REPORT")
    report_lines.append("Apple Financial and Social Analysis Dataset")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: {len(df):,} records, {len(df.columns)} columns")
    report_lines.append(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Overall status
    report_lines.append("\n" + "="*80)
    report_lines.append("OVERALL VALIDATION STATUS")
    report_lines.append("="*80)
    
    overall_status = 'PASS'
    total_checks = 0
    total_warnings = 0
    total_failures = 0
    
    for check_name, result in all_results.items():
        if result['status'] == 'FAIL':
            overall_status = 'FAIL'
        elif result['status'] == 'WARNING' and overall_status != 'FAIL':
            overall_status = 'WARNING'
        
        total_checks += len(result['checks'])
        total_warnings += len(result['warnings'])
        total_failures += len(result['failures'])
    
    report_lines.append(f"\nOverall Status: {overall_status}")
    report_lines.append(f"Total Checks: {total_checks}")
    report_lines.append(f"Warnings: {total_warnings}")
    report_lines.append(f"Failures: {total_failures}")
    
    # Detailed results by category
    for check_name, result in all_results.items():
        report_lines.append("\n" + "="*80)
        report_lines.append(check_name.upper())
        report_lines.append("="*80)
        report_lines.append(f"\nStatus: {result['status']}")
        
        if result['checks']:
            report_lines.append(f"\nPassed Checks ({len(result['checks'])}):")
            for check in result['checks']:
                report_lines.append(check)
        
        if result['warnings']:
            report_lines.append(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result['warnings']:
                report_lines.append(warning)
        
        if result['failures']:
            report_lines.append(f"\nFailures ({len(result['failures'])}):")
            for failure in result['failures']:
                report_lines.append(failure)
        
        # Add suspicious pairs for future correlations
        if 'suspicious_pairs' in result and result['suspicious_pairs']:
            report_lines.append(f"\nSuspicious Feature-Target Pairs ({len(result['suspicious_pairs'])}):")
            for pair in result['suspicious_pairs'][:10]:  # Show top 10
                report_lines.append(f"  • {pair['feature']} -> {pair['target']} "
                                  f"(lag +{pair['lag']}): corr={pair['correlation']:.3f}")
    
    # Summary and recommendations
    report_lines.append("\n" + "="*80)
    report_lines.append("SUMMARY AND RECOMMENDATIONS")
    report_lines.append("="*80)
    
    if overall_status == 'PASS':
        report_lines.append("\n✓ No data leakage detected. Dataset is safe to use for modeling.")
    elif overall_status == 'WARNING':
        report_lines.append("\n⚠ Some suspicious patterns detected. Review warnings before modeling.")
        report_lines.append("\nRecommendations:")
        report_lines.append("  1. Investigate features with high future correlations")
        report_lines.append("  2. Verify rolling window implementations")
        report_lines.append("  3. Cross-validate results with multiple time periods")
    else:
        report_lines.append("\n✗ CRITICAL: Data leakage detected. DO NOT use dataset for modeling.")
        report_lines.append("\nRequired Actions:")
        report_lines.append("  1. Fix target variable alignment issues")
        report_lines.append("  2. Recompute rolling features with proper alignment")
        report_lines.append("  3. Re-validate after fixes")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Report saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Print to console
    print("\n" + "="*80)
    print("VALIDATION REPORT SUMMARY")
    print("="*80)
    print(report_text)


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    """Main execution pipeline."""
    
    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data' / 'processed'
    reports_dir = script_dir / 'reports' / 'final'
    
    input_csv = data_dir / 'apple_with_events.csv'
    output_report = reports_dir / 'leakage_report.txt'
    
    # Load dataset
    df = load_dataset(input_csv)
    
    # Run all validation checks
    all_results = {}
    
    all_results['Lag and Rolling Features'] = check_lag_rolling_features(df)
    all_results['Target Variables'] = check_target_variables(df)
    all_results['Future Correlations'] = check_future_correlations(df)
    all_results['Technical Indicators'] = check_technical_indicators(df)
    all_results['Temporal Consistency'] = check_temporal_consistency(df)
    
    # Generate report
    generate_report(all_results, df, output_report)


if __name__ == '__main__':
    main()
