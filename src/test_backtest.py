#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
df = pd.read_csv('data/processed/apple_feature_enhanced.csv')
df['next_day_return'] = df['close'].shift(-1).pct_change() * 100
df = df.dropna(subset=['next_day_return'])

print('First 10 rows of data:')
print(df[['date', 'close', 'next_day_return']].head(10))
print()

# Check distribution
print(f'Actual returns - Mean: {df["next_day_return"].mean():.4f}%, Std: {df["next_day_return"].std():.4f}%')
print(f'Min: {df["next_day_return"].min():.4f}%, Max: {df["next_day_return"].max():.4f}%')
print()

# Check for extreme values
extreme_returns = df[np.abs(df['next_day_return']) > 10]
print(f'Extreme returns (>10%): {len(extreme_returns)} days')
if len(extreme_returns) > 0:
    print(extreme_returns[['date', 'close', 'next_day_return']])
