"""
package_release.py
==================
Package all project artifacts for Kaggle dataset release.

Creates timestamped release folder with:
- Feature-engineered dataset
- Analysis reports (PDF, CSV)
- Visualizations
- Documentation
- Kaggle metadata JSON

Outputs:
- release_YYYYMMDD_HHMMSS/ directory with all artifacts
- dataset-metadata.json for Kaggle CLI
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    """Get project root directory."""
    current = Path(__file__).resolve().parent
    if current.name == 'src':
        return current.parent
    return current


def create_release_folder():
    """Create timestamped release folder."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    release_name = f'release_{timestamp}'
    
    root = get_project_root()
    release_dir = root / release_name
    release_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[OK] Created release folder: {release_name}")
    return release_dir, release_name


def copy_file_safe(src, dst, description=""):
    """Safely copy file with existence check."""
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        print(f"[WARNING] File not found: {src}")
        return False
    
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        
        size_mb = src.stat().st_size / (1024 * 1024)
        desc = f" ({description})" if description else ""
        print(f"[OK] Copied: {src.name}{desc} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to copy {src}: {e}")
        return False


def copy_directory_safe(src, dst, description="", pattern="*"):
    """Safely copy directory contents."""
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        print(f"[WARNING] Directory not found: {src}")
        return 0
    
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    
    for file in src.glob(pattern):
        if file.is_file():
            try:
                shutil.copy2(file, dst / file.name)
                count += 1
            except Exception as e:
                print(f"[ERROR] Failed to copy {file.name}: {e}")
    
    if count > 0:
        desc = f" ({description})" if description else ""
        print(f"[OK] Copied {count} files from {src.name}{desc}")
    
    return count


def package_datasets(release_dir, root):
    """Package feature-engineered datasets."""
    print("\n[1/7] Packaging datasets...")
    
    data_dir = release_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Main feature dataset
    possible_paths = [
        root / 'data' / 'processed' / 'apple_feature_enhanced.csv',
        root / 'src' / 'data' / 'processed' / 'apple_feature_enhanced.csv'
    ]
    
    for src_path in possible_paths:
        if src_path.exists():
            copy_file_safe(src_path, data_dir / 'apple_feature_enhanced.csv', 
                          'Feature-engineered dataset with 80 columns')
            break


def package_reports(release_dir, root):
    """Package analysis reports."""
    print("\n[2/7] Packaging reports...")
    
    reports_dir = release_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Search in multiple locations
    possible_base_paths = [
        root / 'reports' / 'final',
        root / 'src' / 'reports' / 'final'
    ]
    
    report_files = {
        'final_report.pdf': 'Comprehensive analysis report',
        'final_report.md': 'Markdown report',
        'walkforward_metrics.csv': 'Walk-forward validation metrics',
        'backtest_results.csv': 'Backtest simulation results',
        'equity_daily.csv': 'Daily equity tracking',
        'shap_importance.csv': 'SHAP feature importance',
        'shap_top5.csv': 'Top-5 SHAP features',
        'threshold_grid_results.csv': 'Threshold optimization results',
        'threshold_best.json': 'Best threshold parameters',
        'cost_sensitivity.csv': 'Transaction cost sensitivity',
        'ensemble_weights.json': 'Ensemble model weights',
        'predictions_vs_actual.csv': 'Model predictions',
        'predictions_vs_actual_ensemble.csv': 'Ensemble predictions',
        'shap_dependence_top5.pdf': 'SHAP dependence plots (PDF)'
    }
    
    for base_path in possible_base_paths:
        if base_path.exists():
            for filename, description in report_files.items():
                src = base_path / filename
                if src.exists():
                    copy_file_safe(src, reports_dir / filename, description)


def package_figures(release_dir, root):
    """Package visualization figures."""
    print("\n[3/7] Packaging figures...")
    
    figures_dir = release_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Search in multiple locations
    possible_base_paths = [
        root / 'reports' / 'figures',
        root / 'src' / 'reports' / 'figures'
    ]
    
    for base_path in possible_base_paths:
        if base_path.exists():
            count = copy_directory_safe(base_path, figures_dir, 
                                       'Visualization plots', '*.png')
            if count > 0:
                break


def package_documentation(release_dir, root):
    """Package documentation files."""
    print("\n[4/7] Packaging documentation...")
    
    docs_dir = release_dir / 'docs'
    docs_dir.mkdir(exist_ok=True)
    
    # Documentation files
    doc_files = {
        'variables.csv': 'Feature descriptions',
        'README.md': 'Project documentation',
        'LICENSE': 'License file'
    }
    
    # Search in multiple locations
    search_locations = [
        root / 'docs',
        root / 'src' / 'docs',
        root,
        root / 'src'
    ]
    
    for filename, description in doc_files.items():
        found = False
        for location in search_locations:
            src = location / filename
            if src.exists():
                copy_file_safe(src, docs_dir / filename, description)
                found = True
                break
        
        if not found and filename == 'README.md':
            # Create minimal README if not found
            create_default_readme(docs_dir / 'README.md')


def package_config(release_dir, root):
    """Package configuration files."""
    print("\n[5/7] Packaging configuration...")
    
    config_dir = release_dir / 'config'
    config_dir.mkdir(exist_ok=True)
    
    # Config files
    config_files = [
        'sources.yaml',
        'sources.yml',
        'config.yaml',
        'config.yml'
    ]
    
    # Search in multiple locations
    search_locations = [
        root / 'config',
        root / 'src' / 'config',
        root,
        root / 'src'
    ]
    
    for filename in config_files:
        for location in search_locations:
            src = location / filename
            if src.exists():
                copy_file_safe(src, config_dir / src.name, 'Data sources config')
                break


def create_default_readme(output_path):
    """Create default README if not found."""
    readme_content = """# Apple Financial and Social Analysis Dataset

## Overview

Comprehensive financial dataset for Apple Inc. (AAPL) stock prediction with 80+ engineered features.

## Contents

- **Data**: Feature-engineered dataset with technical indicators, competitor data, and macro variables
- **Reports**: Analysis results including walk-forward validation, backtesting, and SHAP explainability
- **Figures**: Visualization plots for model performance and feature importance
- **Documentation**: Variable descriptions and configuration files

## Features

### Technical Indicators
- RSI, MACD, Bollinger Bands, ATR
- Moving averages, momentum indicators
- Support/resistance levels

### Competitor Data
- Amazon (AMZN), Microsoft (MSFT), Google (GOOGL), Meta (META)
- Correlation analysis with tech sector

### Macroeconomic Variables
- Federal funds rate, CPI, USD index
- Treasury yields, commodity prices
- Economic sentiment indicators

### Social/Search Trends
- Google Trends data for Apple products
- Search interest indices

## Model Performance

- **RMSE**: ~1.93%
- **Walk-Forward Validation**: 10-fold expanding window
- **Ensemble**: LightGBM + XGBoost with optimized blending
- **Sharpe Ratio**: 1.10 (ensemble)

## Usage

See `reports/final_report.pdf` for comprehensive analysis and results.

## License

See LICENSE file for details.

## Citation

If you use this dataset in your research, please cite:

```
Apple Financial and Social Analysis Dataset (2025)
https://www.kaggle.com/datasets/[your-username]/[dataset-slug]
```
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"[OK] Created default README.md")


def create_kaggle_metadata(release_dir, release_name):
    """Create Kaggle dataset metadata JSON."""
    print("\n[6/7] Creating Kaggle metadata...")
    
    metadata = {
        "title": "Apple Stock Prediction - Feature-Engineered Financial Dataset",
        "id": "yourusername/apple-financial-social-analysis",
        "licenses": [
            {
                "name": "CC-BY-SA-4.0"
            }
        ],
        "subtitle": "Comprehensive AAPL dataset with 80+ features: technical indicators, competitors, macro, social trends",
        "description": (
            "# Apple Financial and Social Analysis Dataset\n\n"
            "## Overview\n"
            "Comprehensive financial dataset for Apple Inc. (AAPL) stock analysis with 80+ engineered features "
            "spanning technical indicators, competitor data, macroeconomic variables, and social sentiment.\n\n"
            "## Key Features\n"
            "- **2,696 daily samples** from 2015-03-13 to 2025-11-28\n"
            "- **80 feature columns** including:\n"
            "  - Technical indicators (RSI, MACD, Bollinger Bands, ATR)\n"
            "  - Competitor stocks (AMZN, MSFT, GOOGL, META)\n"
            "  - Macroeconomic data (Fed rate, CPI, USD index, treasuries)\n"
            "  - Social/search trends (Google Trends for Apple products)\n"
            "  - Lagged returns and momentum features\n\n"
            "## Model Performance\n"
            "- **RMSE**: 1.93% (walk-forward validation)\n"
            "- **Ensemble Sharpe**: 1.10 (LightGBM + XGBoost)\n"
            "- **R²**: -0.30 (challenging prediction task)\n\n"
            "## Contents\n"
            "- `data/apple_feature_enhanced.csv` - Main dataset\n"
            "- `reports/` - Analysis results (PDF, CSV)\n"
            "- `figures/` - Visualization plots\n"
            "- `docs/` - Feature descriptions and README\n\n"
            "## Use Cases\n"
            "- Stock price prediction and trading strategies\n"
            "- Time series forecasting with multiple data sources\n"
            "- Feature engineering and selection studies\n"
            "- Ensemble modeling and walk-forward validation\n"
            "- SHAP explainability analysis\n\n"
            "## Citation\n"
            "If you use this dataset, please cite:\n"
            "```\n"
            f"Apple Financial and Social Analysis Dataset ({datetime.now().year})\n"
            "Kaggle Dataset\n"
            "```\n"
        ),
        "keywords": [
            "finance",
            "stocks",
            "time series",
            "machine learning",
            "feature engineering",
            "technical analysis",
            "apple",
            "stock prediction",
            "trading",
            "ensemble modeling"
        ],
        "resources": [
            {
                "path": "data/apple_feature_enhanced.csv",
                "description": "Main feature-engineered dataset with 80+ columns"
            },
            {
                "path": "reports/final_report.pdf",
                "description": "Comprehensive analysis report"
            },
            {
                "path": "reports/shap_importance.csv",
                "description": "SHAP feature importance rankings"
            }
        ]
    }
    
    metadata_path = release_dir / 'dataset-metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Kaggle metadata saved: dataset-metadata.json")
    return metadata_path


def print_release_summary(release_dir, release_name):
    """Print release summary and CLI instructions."""
    print("\n[7/7] Release package summary:")
    print("="*70)
    
    # Count files
    total_files = 0
    total_size = 0
    
    for file in release_dir.rglob('*'):
        if file.is_file():
            total_files += 1
            total_size += file.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n[Release Package]")
    print(f"  Folder:      {release_name}/")
    print(f"  Total Files: {total_files}")
    print(f"  Total Size:  {total_size_mb:.2f} MB")
    print(f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Directory structure
    print(f"\n[Directory Structure]")
    for subdir in ['data', 'reports', 'figures', 'docs', 'config']:
        subdir_path = release_dir / subdir
        if subdir_path.exists():
            file_count = len([f for f in subdir_path.iterdir() if f.is_file()])
            print(f"  {subdir}/ - {file_count} files")
    
    print("\n" + "="*70)
    print("\n[Kaggle CLI Instructions]")
    print("="*70)
    
    print("\n1. First-time publication:")
    print(f"   cd {release_name}")
    print(f"   kaggle datasets create -p .")
    
    print("\n2. Update existing dataset:")
    print(f"   cd {release_name}")
    print(f"   kaggle datasets version -p . -m \"Updated with latest analysis\"")
    
    print("\n3. Before publishing, update dataset-metadata.json:")
    print("   - Change 'yourusername' to your Kaggle username")
    print("   - Customize title/description if needed")
    
    print("\n4. Install Kaggle CLI (if not installed):")
    print("   pip install kaggle")
    print("   kaggle config set -n username YOUR_USERNAME")
    print("   kaggle config set -n key YOUR_API_KEY")
    
    print("\n[Notes]")
    print("- Kaggle API credentials: ~/.kaggle/kaggle.json (Linux/Mac)")
    print("                          C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)")
    print("- Get API key from: https://www.kaggle.com/settings → API → Create New Token")
    
    print("\n" + "="*70)


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("PACKAGE RELEASE - Kaggle Dataset Preparation")
    print("="*70 + "\n")
    
    # Get project root
    root = get_project_root()
    
    # Create release folder
    release_dir, release_name = create_release_folder()
    
    # Package components
    try:
        package_datasets(release_dir, root)
        package_reports(release_dir, root)
        package_figures(release_dir, root)
        package_documentation(release_dir, root)
        package_config(release_dir, root)
        create_kaggle_metadata(release_dir, release_name)
        
        # Summary
        print_release_summary(release_dir, release_name)
        
        print(f"\n[SUCCESS] Release package created successfully!")
        print(f"Location: {release_dir}")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] Failed to create release package: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
