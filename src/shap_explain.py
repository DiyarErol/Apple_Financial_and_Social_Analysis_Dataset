#!/usr/bin/env python3
"""
SHAP Feature Importance & Explanation for Apple Model
======================================================

Analyzes model predictions using SHAP values:
- Loads best_walkforward.pkl and apple_feature_enhanced.csv
- Uses last 500 rows (recent data, no shuffling)
- Computes SHAP values with TreeExplainer for LightGBM
- Outputs:
    - reports/final/shap_importance.csv (mean |SHAP| per feature)
    - reports/figures/shap_summary.png (summary plot)
    - reports/figures/shap_dependence_top5.png (top 5 features)
- Prints top-10 influential features
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports' / 'final'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
MODELS_DIR = BASE_DIR / 'models'

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
np.random.seed(42)


def check_shap_installation():
    """Check if SHAP is installed, provide guidance if not."""
    try:
        import shap
        return shap
    except ImportError:
        print("\n[ERROR] SHAP library not found!")
        print("\nTo install SHAP, run:")
        print("  pip install shap")
        print("\nOr if using conda:")
        print("  conda install -c conda-forge shap")
        print("\nSHAP is required for model explainability analysis.")
        sys.exit(1)


def load_model_and_data():
    """Load model and dataset."""
    
    print("[1/6] Loading model and dataset...")
    
    # Load model
    model_path = MODELS_DIR / 'best_walkforward.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    model = model_dict['model']
    scaler = model_dict.get('scaler', None)
    feature_cols = model_dict['feature_cols']
    
    print(f"[OK] Model loaded: {type(model).__name__}")
    print(f"[OK] Features: {len(feature_cols)}")
    print(f"[OK] Scaler: {'Yes' if scaler else 'No'}")
    
    # Load dataset
    csv_path = DATA_DIR / 'apple_feature_enhanced.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[OK] Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    return model, scaler, feature_cols, df


def prepare_feature_matrix(df, feature_cols, scaler):
    """Recreate exact feature matrix X used by model."""
    
    print("[2/6] Preparing feature matrix...")
    
    # Add missing columns with zeros
    for col in feature_cols:
        if col not in df.columns:
            print(f"[WARN] Missing feature '{col}', filling with 0")
            df[col] = 0
    
    # Extract features (exclude date and targets)
    X = df[feature_cols].copy()
    
    # Fill NaN values
    X = X.fillna(0)
    
    print(f"[OK] Feature matrix prepared: {X.shape}")
    
    # Apply scaling if scaler exists
    if scaler is not None:
        X_scaled = scaler.transform(X)
        print("[OK] Features scaled")
        return X_scaled, X  # Return both scaled and unscaled
    else:
        return X.values, X


def select_recent_slice(X, X_unscaled, n_samples=500):
    """Select last n_samples rows (recent data, no shuffling)."""
    
    print(f"[3/6] Selecting recent {n_samples} samples...")
    
    if len(X) <= n_samples:
        print(f"[INFO] Using all {len(X)} samples (dataset smaller than {n_samples})")
        return X, X_unscaled
    
    # Take last n_samples rows
    X_recent = X[-n_samples:]
    X_unscaled_recent = X_unscaled.iloc[-n_samples:]
    
    print(f"[OK] Selected {len(X_recent)} recent samples")
    
    return X_recent, X_unscaled_recent


def compute_shap_values(model, X, shap):
    """Compute SHAP values using TreeExplainer."""
    
    print("[4/6] Computing SHAP values...")
    
    try:
        # Create explainer for tree-based models (LightGBM)
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X)
        
        print(f"[OK] SHAP values computed: shape {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
        
        return explainer, shap_values
    
    except Exception as e:
        print(f"[ERROR] Failed to compute SHAP values: {e}")
        sys.exit(1)


def save_shap_importance(shap_values, feature_cols, output_path):
    """Calculate and save mean |SHAP| per feature."""
    
    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create dataframe
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Save to CSV
    shap_importance.to_csv(output_path, index=False)
    
    print(f"[OK] SHAP importance saved: {output_path}")
    
    return shap_importance


def plot_shap_summary(shap_values, X_unscaled, feature_cols, output_path, shap):
    """Generate SHAP summary plot."""
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_unscaled, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] SHAP summary plot saved: {output_path}")


def plot_shap_dependence_top5(shap_values, X_unscaled, feature_cols, top5_features, output_path, shap):
    """Generate SHAP dependence plots for top 5 features (PNG grid)."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(top5_features):
        if i >= 5:
            break
        
        plt.sca(axes[i])
        shap.dependence_plot(
            feat, 
            shap_values, 
            X_unscaled, 
            feature_names=feature_cols, 
            show=False,
            ax=axes[i]
        )
        axes[i].set_title(f'SHAP Dependence: {feat}', fontsize=12, fontweight='bold')
    
    # Hide unused subplot
    if len(top5_features) < 6:
        axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] SHAP dependence plot (top 5) saved: {output_path}")


def plot_shap_dependence_multipage_pdf(shap_values, X_unscaled, feature_cols, top5_features, output_path, shap):
    """Generate multipage PDF with one dependence plot per page for top 5 features."""
    
    print(f"[INFO] Creating multipage PDF for top-5 features...")
    
    with PdfPages(output_path) as pdf:
        for i, feat in enumerate(top5_features, 1):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            try:
                shap.dependence_plot(
                    feat,
                    shap_values,
                    X_unscaled,
                    feature_names=feature_cols,
                    show=False,
                    ax=ax
                )
                
                ax.set_title(f'SHAP Dependence Plot: {feat}\n(Rank #{i})', 
                           fontsize=14, fontweight='bold', pad=20)
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"[WARNING] Failed to create dependence plot for {feat}: {e}")
                plt.close(fig)
                continue
        
        # Add metadata page
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        metadata_text = (
            "SHAP Dependence Analysis - Top 5 Features\n\n"
            "This PDF contains individual dependence plots for the 5 most\n"
            "influential features based on mean absolute SHAP values.\n\n"
            "Features Analyzed:\n"
        )
        
        for i, feat in enumerate(top5_features, 1):
            metadata_text += f"  {i}. {feat}\n"
        
        metadata_text += (
            "\n\nInterpretation:\n"
            "- Y-axis: SHAP value (impact on model output)\n"
            "- X-axis: Feature value\n"
            "- Color: Interaction with another feature\n"
            "- Higher SHAP value = stronger positive impact on prediction\n"
        )
        
        ax.text(0.5, 0.5, metadata_text, 
               ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"[OK] Multipage PDF saved: {output_path}")


def save_top5_shap_csv(shap_importance, output_path):
    """Save top-5 features to dedicated CSV."""
    
    top5 = shap_importance.head(5).copy()
    top5.reset_index(drop=True, inplace=True)
    top5.insert(0, 'rank', range(1, 6))
    
    top5.to_csv(output_path, index=False)
    print(f"[OK] Top-5 features CSV saved: {output_path}")
    
    return top5


def print_top_features(shap_importance, n=10):
    """Print top N influential features."""
    
    print(f"\n{'='*70}")
    print(f"TOP {n} INFLUENTIAL FEATURES (Mean |SHAP|)")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Feature':<35} {'Mean |SHAP|':<15}")
    print(f"{'-'*70}")
    
    for i, row in shap_importance.head(n).iterrows():
        rank = i + 1 if isinstance(i, int) else shap_importance.index.get_loc(i) + 1
        print(f"{rank:<6} {row['feature']:<35} {row['mean_abs_shap']:<15.6f}")
    
    print(f"{'='*70}\n")


def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("SHAP EXPLAINABILITY ANALYSIS - Apple Stock Prediction Model")
    print("="*70 + "\n")
    
    try:
        # Check SHAP installation
        shap = check_shap_installation()
        print("[OK] SHAP library found\n")
        
        # Load model and data
        model, scaler, feature_cols, df = load_model_and_data()
        
        # Prepare feature matrix
        X_scaled, X_unscaled = prepare_feature_matrix(df, feature_cols, scaler)
        
        # Select recent slice
        X_recent, X_unscaled_recent = select_recent_slice(X_scaled, X_unscaled, n_samples=500)
        
        # Compute SHAP values
        explainer, shap_values = compute_shap_values(model, X_recent, shap)
        
        # Save outputs
        print("[5/6] Saving outputs...")
        
        # SHAP importance CSV (all features)
        shap_importance_path = REPORTS_DIR / 'shap_importance.csv'
        shap_importance = save_shap_importance(shap_values, feature_cols, shap_importance_path)
        
        # Get top-5 features
        top5_features = shap_importance.head(5)['feature'].tolist()
        
        # Top-5 dedicated CSV
        top5_csv_path = REPORTS_DIR / 'shap_top5.csv'
        save_top5_shap_csv(shap_importance, top5_csv_path)
        
        # SHAP summary plot
        summary_plot_path = FIGURES_DIR / 'shap_summary.png'
        plot_shap_summary(shap_values, X_unscaled_recent, feature_cols, summary_plot_path, shap)
        
        # SHAP dependence plots - PNG grid (top 5)
        dependence_plot_path = FIGURES_DIR / 'shap_dependence_top5.png'
        plot_shap_dependence_top5(shap_values, X_unscaled_recent, feature_cols, 
                                   top5_features, dependence_plot_path, shap)
        
        # SHAP dependence plots - Multipage PDF (top 5)
        dependence_pdf_path = REPORTS_DIR / 'shap_dependence_top5.pdf'
        plot_shap_dependence_multipage_pdf(shap_values, X_unscaled_recent, feature_cols,
                                          top5_features, dependence_pdf_path, shap)
        
        # Print top features
        print("[6/6] Analysis complete!")
        print_top_features(shap_importance, n=10)
        
        print("[SUCCESS] SHAP explainability analysis completed!")
        print(f"\nOutputs:")
        print(f"  - {shap_importance_path} (all features)")
        print(f"  - {top5_csv_path} (top-5 features)")
        print(f"  - {summary_plot_path} (summary visualization)")
        print(f"  - {dependence_plot_path} (dependence grid PNG)")
        print(f"  - {dependence_pdf_path} (dependence multipage PDF)")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
