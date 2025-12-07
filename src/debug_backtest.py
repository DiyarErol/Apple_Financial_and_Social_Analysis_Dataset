#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
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

# Generate predictions
y_pred = model.predict(X_scaled)

print("Prediction Statistics:")
print(f"Mean: {np.mean(y_pred):.6f}%")
print(f"Std: {np.std(y_pred):.6f}%")
print(f"Min: {np.min(y_pred):.6f}%")
print(f"Max: {np.max(y_pred):.6f}%")
print()

# Check first 20 predictions vs actuals
print("First 20 predictions vs actuals:")
print(f"{'Actual':>10} {'Predicted':>10} {'Position':>10}")
print("-" * 35)

LONG_THRESHOLD = 0.5
SHORT_THRESHOLD = -0.5

for i in range(min(20, len(df))):
    actual = df['next_day_return'].iloc[i]
    predicted = y_pred[i]
    
    if predicted > LONG_THRESHOLD:
        position = 'LONG'
    elif predicted < SHORT_THRESHOLD:
        position = 'SHORT'
    else:
        position = 'FLAT'
    
    print(f"{actual:>10.4f} {predicted:>10.4f} {position:>10}")

print()

# Check how often we're going long/short
long_count = np.sum(y_pred > LONG_THRESHOLD)
short_count = np.sum(y_pred < SHORT_THRESHOLD)
flat_count = len(y_pred) - long_count - short_count

print(f"Position distribution:")
print(f"Long:  {long_count:>5} ({long_count/len(y_pred)*100:.2f}%)")
print(f"Short: {short_count:>5} ({short_count/len(y_pred)*100:.2f}%)")
print(f"Flat:  {flat_count:>5} ({flat_count/len(y_pred)*100:.2f}%)")
print()

# Check correlation between predicted and actual returns
correlation = np.corrcoef(y_pred, df['next_day_return'])[0, 1]
print(f"Correlation (predicted vs actual): {correlation:.6f}")
