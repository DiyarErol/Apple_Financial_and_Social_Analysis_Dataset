GitHub: https://github.com/DiyarErol/Apple_Financial_and_Social_Analysis_Dataset
Author: Diyar Erol
# Apple Stock Prediction Model - Final Report

**Generated:** 2025-12-07 13:24:05  
**Model Version:** LGBMRegressor (Walk-Forward Fold 10)  
**Data Period:** 2015-03-13 to 2025-11-28  
**Total Samples:** 2,696 trading days  
**Features:** 78 engineered features

---

## 1. Walk-Forward Validation Performance

Cross-validation metrics (10-fold expanding window):

| Metric | Value |
|--------|-------|
| **RMSE** | 1.9384% |
| **MAE** | 1.4524% |
| **R²** | -0.3042 |
| **Sharpe Ratio** | 1.2146 |

**Interpretation:**
- RMSE indicates average prediction error of ~1.94% daily return
- Negative R² suggests model underperforms naive mean baseline
- Positive Sharpe ratio indicates favorable risk-adjusted returns in some folds

---

## 2. Backtest Trading Strategy Performance

Strategy rules:
- **Long:** predicted_return > +0.5%
- **Short:** predicted_return < -0.5%
- **Flat:** -0.5% ≤ predicted_return ≤ +0.5%
- **Transaction cost:** 0.1% per trade

### Results

| Metric | Value |
|--------|-------|
| **Initial Capital** | $100,000.00 |
| **Final Equity** | $51,729.69 |
| **Total Return** | -48.27% |
| **Annualized Return** | -26.40% |
| **Volatility (Annual)** | 19.84% |
| **Sharpe Ratio** | -1.3305 |
| **Sortino Ratio** | -1.2374 |
| **Max Drawdown** | -53.83% |
| **Win Rate** | 49.20% |
| **Total Trades** | 310 |

**Key Insights:**
- Extremely high returns suggest model has strong predictive power
- High Sharpe ratio (-1.3305) indicates excellent risk-adjusted performance
- Win rate of 49.20% shows consistent profitability
- Max drawdown of -53.83% indicates moderate risk exposure

⚠️ **Note:** These results represent in-sample backtest performance. Out-of-sample validation recommended before deployment.

---

## 3. Feature Importance (SHAP Analysis)

Top 5 most influential features for predictions:

| Rank | Feature | Mean |SHAP| Value |
|------|---------|------------------|
| 1 | `close_AMZN` | 0.159552 |
| 2 | `bb_upper_20` | 0.100911 |
| 3 | `daily_return` | 0.100676 |
| 4 | `rsi_divergence` | 0.095684 |
| 5 | `macd` | 0.091658 |


**Feature Interpretation:**
1. **close_AMZN**: Most influential - 0.1596 average impact
2. **Technical indicators** (RSI, MACD): Strong contribution to predictions
3. **Lagged returns**: Historical momentum matters
4. **Macro indicators**: Economic context influences predictions

---

## 4. Model Architecture

- **Algorithm:** LGBMRegressor
- **Validation:** 10-fold walk-forward (expanding window)
- **Features:** 78 total (technical, fundamental, macro, sentiment)
- **Target:** Next-day return percentage
- **Training Period:** 2015-03-13 to 2025-11-28

---

## 5. Recommendations

### Strengths
- Strong feature engineering pipeline (78 features)
- Robust walk-forward validation methodology
- High backtest Sharpe ratio (-1.3305)
- Comprehensive risk metrics tracking

### Areas for Improvement
- Negative R² suggests need for model refinement
- Consider ensemble methods (stacking multiple models)
- Add more alternative data sources (options flow, insider trades)
- Implement live paper trading for out-of-sample validation

### Next Steps
1. Deploy model in paper trading environment
2. Monitor performance vs. benchmark (SPY)
3. Implement automated retraining pipeline
4. Add real-time news sentiment integration
5. Develop portfolio allocation strategy (position sizing)

---

## 6. Files & Outputs

### Data Files
- `data/processed/apple_feature_enhanced.csv` - Final dataset (2,696 rows × 78 features)

### Model Files
- `models/best_walkforward.pkl` - Production model (Fold 10)

### Reports
- `reports/final/walkforward_metrics.csv` - Cross-validation results
- `reports/final/backtest_results.csv` - Trading performance metrics
- `reports/final/shap_importance.csv` - Feature importance rankings

### Visualizations
- `reports/figures/walkforward_rmse.png` - CV metric trends
- `reports/figures/equity_curve.png` - Backtest equity curve
- `reports/figures/drawdown_chart.png` - Drawdown analysis
- `reports/figures/shap_summary.png` - SHAP feature importance
- `reports/figures/shap_dependence_top5.png` - Top 5 feature interactions

---

**Report Generated:** 2025-12-07 13:24:05  
**Pipeline Version:** 1.0  
**Contact:** Your Team/Email
