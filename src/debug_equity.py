#!/usr/bin/env python3
"""Debug the backtest equity calculation."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Load data
df = pd.read_csv(DATA_DIR / 'apple_feature_enhanced.csv')
df['next_day_return'] = df['close'].shift(-1).pct_change() * 100
df = df.dropna(subset=['next_day_return'])

# Load model
with open(MODELS_DIR / 'best_walkforward.pkl', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
scaler = model_dict['scaler']
feature_cols = model_dict['feature_cols']

# Prepare features
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

X = df[feature_cols].fillna(0)
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Check first 30 rows
print("First 30 days analysis:")
print("{:>5} {:>10} {:>10} {:>8} {:>15}".format("Day", "Actual%", "Pred%", "Position", "Equity"))
print("-" * 60)

equity = 100000
position = 0
LONG_THRESHOLD = 0.5
SHORT_THRESHOLD = -0.5

for i in range(30):
    actual = df['next_day_return'].iloc[i]
    predicted = y_pred[i]
    
    # Determine new position
    if predicted > LONG_THRESHOLD:
        new_position = 1
    elif predicted < SHORT_THRESHOLD:
        new_position = -1
    else:
        new_position = 0
    
    # Apply transaction cost if position changed
    if i > 0 and position != new_position:
        trans_cost = 0.001
        transaction_cost_pct = trans_cost
        if new_position != 0:
            transaction_cost_pct += trans_cost
        equity = equity * (1 - transaction_cost_pct)
    
    # Apply strategy return
    strategy_return = new_position * (actual / 100)
    equity = equity * (1 + strategy_return)
    
    pos_str = 'LONG' if new_position == 1 else ('SHORT' if new_position == -1 else 'FLAT')
    print("{:>5} {:>10.4f} {:>10.4f} {:>8} {:>15.2f}".format(i+1, actual, predicted, pos_str, equity))
    
    position = new_position

print("\nEquity after 30 days: ${:,.2f}".format(equity))
print("\nCheck for leverage or compound effect:")
print("Daily multiplier check:")
for i in range(5):
    actual = df['next_day_return'].iloc[i]
    multiplier = 1 + actual / 100
    print("  Day {}: 1 + {:.4f}% = {:.6f}".format(i+1, actual, multiplier))
