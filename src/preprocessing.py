"""Data preprocessing module for Apple Financial and Social Analysis Dataset."""

import pandas as pd
import numpy as np


def clean_data(df):
    """Clean and prepare data for analysis."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna(subset=['ticker', 'date'])
    
    # Convert data types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def normalize_features(df, columns=None):
    """Normalize numerical features."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df
