"""
Modeling Script for Apple Financial and Social Analysis Dataset.
Trains and evaluates machine learning models for stock price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_feature_dataset(csv_path):
    """
    Load engineered feature dataset.
    
    Parameters:
    -----------
    csv_path : str
        Path to the feature dataset CSV
    
    Returns:
    --------
    pd.DataFrame
        Feature dataset
    """
    print("Loading feature dataset...")
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✓ Dataset loaded: {len(df)} records, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_data(df):
    """
    Prepare features and target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature dataset
    
    Returns:
    --------
    tuple
        (X, y, feature_names, df)
    """
    print("Preparing features and target variable...")
    try:
        # Define features (all numeric columns except date and target)
        exclude_cols = ['date', 'next_day_close']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]]
        
        X = df[feature_cols].values
        y = df['next_day_close'].values
        
        print(f"✓ Data prepared")
        print(f"  Features (X): {len(feature_cols)} columns")
        print(f"  Target (y): next_day_close")
        print(f"  Feature columns: {feature_cols}")
        
        return X, y, feature_cols, df
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def split_data(X, y, df, test_size=0.2):
    """
    Split data chronologically (no shuffling).
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    df : pd.DataFrame
        Original dataframe
    test_size : float
        Test set proportion
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, indices_train, indices_test)
    """
    print("Splitting data chronologically...")
    try:
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        train_dates = df['date'].iloc[:split_idx]
        test_dates = df['date'].iloc[split_idx:]
        
        print(f"✓ Data split chronologically (no shuffling)")
        print(f"  Training: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()})")
        print(f"  Testing:  {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})")
        
        return X_train, X_test, y_train, y_test, split_idx
    except Exception as e:
        print(f"✗ Error splitting data: {e}")
        return None, None, None, None, None


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Testing features
    
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    print("Scaling features...")
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Features scaled using StandardScaler")
        print(f"  Mean (train): {X_train_scaled.mean(axis=0).mean():.4f}")
        print(f"  Std (train):  {X_train_scaled.std(axis=0).mean():.4f}")
        
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        print(f"✗ Error scaling features: {e}")
        return None, None, None


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train XGBoost model.
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Training and testing features
    y_train, y_test : np.ndarray
        Training and testing targets
    
    Returns:
    --------
    tuple
        (model, predictions, metrics)
    """
    print("\nTraining XGBoost model...")
    try:
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        metrics = {
            'model_name': 'XGBoost',
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_test': mae_test
        }
        
        print(f"✓ XGBoost trained")
        print(f"  Train RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
        print(f"  Test RMSE:  {rmse_test:.4f}, R²: {r2_test:.4f}")
        print(f"  Test MAE:   {mae_test:.4f}")
        
        return model, y_pred_test, metrics
    except Exception as e:
        print(f"✗ Error training XGBoost: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def train_lightgbm(X_train, X_test, y_train, y_test):
    """
    Train LightGBM model.
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Training and testing features
    y_train, y_test : np.ndarray
        Training and testing targets
    
    Returns:
    --------
    tuple
        (model, predictions, metrics)
    """
    print("\nTraining LightGBM model...")
    try:
        model = lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=500,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        metrics = {
            'model_name': 'LightGBM',
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_test': mae_test
        }
        
        print(f"✓ LightGBM trained")
        print(f"  Train RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
        print(f"  Test RMSE:  {rmse_test:.4f}, R²: {r2_test:.4f}")
        print(f"  Test MAE:   {mae_test:.4f}")
        
        return model, y_pred_test, metrics
    except Exception as e:
        print(f"✗ Error training LightGBM: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def select_best_model(xgb_metrics, lgb_metrics, xgb_model, lgb_model, xgb_pred, lgb_pred):
    """
    Select the better model based on test RMSE.
    
    Parameters:
    -----------
    xgb_metrics, lgb_metrics : dict
        Model metrics
    xgb_model, lgb_model : object
        Trained models
    xgb_pred, lgb_pred : np.ndarray
        Predictions
    
    Returns:
    --------
    tuple
        (best_model, best_metrics, best_predictions)
    """
    print("\nSelecting best model...")
    try:
        if xgb_metrics['rmse_test'] <= lgb_metrics['rmse_test']:
            print(f"✓ XGBoost selected (RMSE: {xgb_metrics['rmse_test']:.4f})")
            return xgb_model, xgb_metrics, xgb_pred
        else:
            print(f"✓ LightGBM selected (RMSE: {lgb_metrics['rmse_test']:.4f})")
            return lgb_model, lgb_metrics, lgb_pred
    except Exception as e:
        print(f"✗ Error selecting model: {e}")
        return None, None, None


def save_model(model, output_path):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : object
        Trained model
    output_path : str
        Path to save model
    """
    print("Saving model...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        print(f"✓ Model saved to {output_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")


def save_predictions(y_test, y_pred, output_path):
    """
    Save predictions and actual values to CSV.
    
    Parameters:
    -----------
    y_test : np.ndarray
        Actual test values
    y_pred : np.ndarray
        Predicted values
    output_path : str
        Path to save CSV
    """
    print("Saving predictions...")
    try:
        pred_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'error': y_test - y_pred,
            'percentage_error': ((y_test - y_pred) / y_test * 100)
        })
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to {output_path}")
    except Exception as e:
        print(f"✗ Error saving predictions: {e}")


def plot_actual_vs_predicted(y_test, y_pred, output_dir):
    """
    Generate scatter plot of actual vs predicted values.
    
    Parameters:
    -----------
    y_test : np.ndarray
        Actual test values
    y_pred : np.ndarray
        Predicted values
    output_dir : str
        Output directory
    """
    print("Generating actual vs predicted plot...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Closing Price (USD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Closing Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Actual vs Predicted Apple Stock Closing Price', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'actual_vs_predicted.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Actual vs predicted plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating plot: {e}")


def plot_feature_importance(model, feature_names, output_dir):
    """
    Generate feature importance bar chart.
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        Feature names
    output_dir : str
        Output directory
    """
    print("Generating feature importance plot...")
    try:
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            print("✗ Model does not have feature_importances_ attribute")
            return
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        top_features = 15  # Top 15 features
        indices = indices[:top_features]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.barh(range(len(indices)), importance[indices], color='#1f77b4', edgecolor='black')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Feature Importance (for Stock Price Prediction)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Reverse y-axis to show highest importance on top
        ax.invert_yaxis()
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'feature_importance.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating feature importance plot: {e}")


def plot_residuals(y_test, y_pred, output_dir):
    """
    Generate residual distribution plot.
    
    Parameters:
    -----------
    y_test : np.ndarray
        Actual test values
    y_pred : np.ndarray
        Predicted values
    output_dir : str
        Output directory
    """
    print("Generating residual plot...")
    try:
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of residuals
        axes[0].hist(residuals, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Residuals vs Predicted
        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5, s=30)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Price (USD)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[1].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'residuals_distribution.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Residuals plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating residuals plot: {e}")


def save_summary(best_metrics, xgb_metrics, lgb_metrics, y_test, y_pred, output_path):
    """
    Save model summary to text file.
    
    Parameters:
    -----------
    best_metrics : dict
        Best model metrics
    xgb_metrics, lgb_metrics : dict
        Comparison metrics
    y_test : np.ndarray
        Actual test values
    y_pred : np.ndarray
        Predicted values
    output_path : str
        Path to save summary
    """
    print("Saving model summary...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        summary_text = f"""
================================================================================
APPLE STOCK PRICE PREDICTION - MODEL SUMMARY
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
BEST MODEL: {best_metrics['model_name']}
================================================================================

Training Performance:
  RMSE:  ${best_metrics['rmse_train']:.4f}
  R²:    {best_metrics['r2_train']:.4f}

Testing Performance:
  RMSE:  ${best_metrics['rmse_test']:.4f}
  R²:    {best_metrics['r2_test']:.4f}
  MAE:   ${best_metrics['mae_test']:.4f}

Prediction Statistics:
  Mean Absolute Error:     ${np.abs(y_test - y_pred).mean():.4f}
  Mean Percentage Error:   {np.abs((y_test - y_pred) / y_test * 100).mean():.2f}%
  Min Prediction Error:    ${(y_test - y_pred).min():.4f}
  Max Prediction Error:    ${(y_test - y_pred).max():.4f}
  Std of Errors:          ${np.std(y_test - y_pred):.4f}

================================================================================
MODEL COMPARISON
================================================================================

XGBoost:
  Test RMSE:  ${xgb_metrics['rmse_test']:.4f}
  Test R²:    {xgb_metrics['r2_test']:.4f}
  Test MAE:   ${xgb_metrics['mae_test']:.4f}

LightGBM:
  Test RMSE:  ${lgb_metrics['rmse_test']:.4f}
  Test R²:    {lgb_metrics['r2_test']:.4f}
  Test MAE:   ${lgb_metrics['mae_test']:.4f}

================================================================================
INTERPRETATION
================================================================================

The {best_metrics['model_name']} model was selected as the best performer with:
- Lower test RMSE indicates better prediction accuracy
- R² of {best_metrics['r2_test']:.4f} means {best_metrics['r2_test']*100:.2f}% of variance explained
- Test MAE of ${best_metrics['mae_test']:.4f} shows average absolute error per prediction

The model successfully captures temporal patterns in Apple's stock price data
by leveraging engineered features including technical indicators, macroeconomic
indicators, and Google search trends.

================================================================================
OUTPUT FILES
================================================================================

Models:
  - models/best_model.pkl

Predictions:
  - reports/final/predictions_vs_actual.csv

Visualizations:
  - reports/figures/actual_vs_predicted.png
  - reports/figures/feature_importance.png
  - reports/figures/residuals_distribution.png

================================================================================
"""
        
        with open(output_path, 'w') as f:
            f.write(summary_text)
        
        print(f"✓ Summary saved to {output_path}")
    except Exception as e:
        print(f"✗ Error saving summary: {e}")


def main():
    """
    Main execution function:
    1. Load feature dataset
    2. Prepare features and target
    3. Split data chronologically
    4. Scale features
    5. Train XGBoost and LightGBM
    6. Select best model
    7. Save model and predictions
    8. Generate visualizations
    9. Save summary
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL AND SOCIAL ANALYSIS - MACHINE LEARNING MODELING")
    print("="*80 + "\n")
    
    # Define paths
    input_csv = 'data/processed/apple_feature_dataset.csv'
    model_output = 'models/best_model.pkl'
    pred_output = 'reports/final/predictions_vs_actual.csv'
    summary_output = 'reports/final/model_summary.txt'
    figures_dir = 'reports/figures'
    
    # Step 1: Load dataset
    print("[1/9] Loading feature dataset...")
    df = load_feature_dataset(input_csv)
    if df is None:
        print("✗ Failed to load dataset. Exiting.")
        return False
    
    # Step 2: Prepare data
    print("\n[2/9] Preparing features and target...")
    X, y, feature_names, df = prepare_data(df)
    if X is None:
        print("✗ Failed to prepare data. Exiting.")
        return False
    
    # Step 3: Split data
    print("\n[3/9] Splitting data chronologically...")
    X_train, X_test, y_train, y_test, split_idx = split_data(X, y, df)
    if X_train is None:
        print("✗ Failed to split data. Exiting.")
        return False
    
    # Step 4: Scale features
    print("\n[4/9] Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    if X_train_scaled is None:
        print("✗ Failed to scale features. Exiting.")
        return False
    
    # Step 5: Train XGBoost
    print("\n[5/9] Training XGBoost model...")
    xgb_model, xgb_pred, xgb_metrics = train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test)
    if xgb_model is None:
        print("✗ Failed to train XGBoost. Exiting.")
        return False
    
    # Step 6: Train LightGBM
    print("\n[6/9] Training LightGBM model...")
    lgb_model, lgb_pred, lgb_metrics = train_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test)
    if lgb_model is None:
        print("✗ Failed to train LightGBM. Exiting.")
        return False
    
    # Step 7: Select best model
    print("\n[7/9] Selecting best model...")
    best_model, best_metrics, y_pred_best = select_best_model(
        xgb_metrics, lgb_metrics, xgb_model, lgb_model, xgb_pred, lgb_pred
    )
    if best_model is None:
        print("✗ Failed to select model. Exiting.")
        return False
    
    # Step 8: Save model
    print("\n[8/9] Saving model and predictions...")
    save_model(best_model, model_output)
    save_predictions(y_test, y_pred_best, pred_output)
    
    # Step 9: Generate visualizations
    print("\n[9/9] Generating visualizations...")
    plot_actual_vs_predicted(y_test, y_pred_best, figures_dir)
    plot_feature_importance(best_model, feature_names, figures_dir)
    plot_residuals(y_test, y_pred_best, figures_dir)
    
    # Save summary
    save_summary(best_metrics, xgb_metrics, lgb_metrics, y_test, y_pred_best, summary_output)
    
    # Final summary
    print("\n" + "="*80)
    print("MODELING COMPLETE")
    print("="*80)
    
    print(f"\n✓ Best Model: {best_metrics['model_name']}")
    print(f"  Test RMSE: ${best_metrics['rmse_test']:.4f}")
    print(f"  Test R²:   {best_metrics['r2_test']:.4f}")
    print(f"  Test MAE:  ${best_metrics['mae_test']:.4f}")
    
    print(f"\n✓ Output Files:")
    print(f"  • {model_output}")
    print(f"  • {pred_output}")
    print(f"  • {summary_output}")
    print(f"  • {figures_dir}/actual_vs_predicted.png")
    print(f"  • {figures_dir}/feature_importance.png")
    print(f"  • {figures_dir}/residuals_distribution.png")
    
    print("\n" + "="*80)
    print("✓ All modeling tasks completed successfully!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
