"""Visualization module for Apple Financial and Social Analysis Dataset."""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_time_series(df, column, title, output_path=None):
    """Plot time series data."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column], linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True, alpha=0.3)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_heatmap(df, output_path=None):
    """Plot correlation heatmap of numerical features."""
    plt.figure(figsize=(10, 8))
    correlation = df.corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
