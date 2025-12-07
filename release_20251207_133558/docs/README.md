# Apple Financial and Social Analysis Dataset

## Overview
This project analyzes Apple's financial metrics and correlates them with social media sentiment data to identify patterns and insights.

## Project Structure

```
Apple_Financial_and_Social_Analysis_Dataset/
├── data/
│   ├── raw/                 # Raw data files
│   ├── processed/           # Cleaned and processed data
│   └── external/            # External data sources
├── reports/
│   ├── figures/             # Generated visualizations
│   └── final/               # Final analysis reports
├── src/
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessing.py     # Data cleaning and preprocessing
│   ├── visualization.py     # Plotting and visualization functions
│   └── analysis.py          # Analysis and statistical functions
├── config/
│   └── settings.yaml        # Configuration settings
├── requirements.txt         # Python package dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Loading Data
```python
from src.data_loader import load_raw_data, save_processed_data

# Load data
df = load_raw_data('data/raw/apple_stock_data.csv')

# Process and save
df_processed = preprocess_data(df)
save_processed_data(df_processed, 'data/processed/apple_stock_processed.csv')
```

### Data Preprocessing
```python
from src.preprocessing import clean_data, normalize_features

df_clean = clean_data(df)
df_normalized = normalize_features(df_clean)
```

### Analysis
```python
from src.analysis import calculate_statistics, analyze_trends

stats = calculate_statistics(df)
trends = analyze_trends(df)
```

### Visualization
```python
from src.visualization import plot_time_series, plot_correlation_heatmap

plot_time_series(df, 'close_price', 'Apple Stock Price Over Time',
                 'reports/figures/price_trend.png')
plot_correlation_heatmap(df, 'reports/figures/correlation.png')
```

## Configuration

Edit `config/settings.yaml` to customize:
- Data paths
- Column names
- Visualization settings
- Analysis parameters

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pyyaml
- requests

## License

MIT License

## Author

Data Analysis Project
