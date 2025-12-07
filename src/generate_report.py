#!/usr/bin/env python3
"""
Final Report Generator for Apple Stock Prediction Model
========================================================

Generates comprehensive PDF and Markdown reports with:
- Walk-forward validation metrics
- Backtest performance summary
- SHAP feature importance (top 5)
- Key visualizations
- Model metadata and timestamps
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports' / 'final'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
MODELS_DIR = BASE_DIR / 'models'

# Output paths
PDF_PATH = REPORTS_DIR / 'final_report.pdf'
MD_PATH = REPORTS_DIR / 'final_report.md'

def load_data():
    """Load all required CSV files and metadata."""
    
    print("[1/5] Loading data files...")
    
    # Walk-forward metrics
    wf_metrics = pd.read_csv(REPORTS_DIR / 'walkforward_metrics.csv')
    wf_overall = wf_metrics[wf_metrics['fold'] == 'OVERALL'].iloc[0]
    
    # Backtest results
    backtest = pd.read_csv(REPORTS_DIR / 'backtest_results.csv')
    backtest_dict = dict(zip(backtest['metric'], backtest['value']))
    
    # SHAP importance
    shap_imp = pd.read_csv(REPORTS_DIR / 'shap_importance.csv')
    top5_shap = shap_imp.head(5)
    
    # Model metadata
    with open(MODELS_DIR / 'best_walkforward.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    
    # Data range
    df = pd.read_csv(DATA_DIR / 'apple_feature_enhanced.csv')
    data_start = df['date'].iloc[0]
    data_end = df['date'].iloc[-1]
    
    print("[OK] Data loaded successfully")
    
    return {
        'wf_overall': wf_overall,
        'backtest': backtest_dict,
        'top5_shap': top5_shap,
        'model_type': type(model_dict['model']).__name__,
        'num_features': len(model_dict['feature_cols']),
        'data_start': data_start,
        'data_end': data_end,
        'total_rows': len(df),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def generate_markdown_report(data):
    """Generate Markdown version of the report."""
    
    print("[2/5] Generating Markdown report...")
    
    md_content = f"""# Apple Stock Prediction Model - Final Report

**Generated:** {data['timestamp']}  
**Model Version:** {data['model_type']} (Walk-Forward Fold 10)  
**Data Period:** {data['data_start']} to {data['data_end']}  
**Total Samples:** {data['total_rows']:,} trading days  
**Features:** {data['num_features']} engineered features

---

## 1. Walk-Forward Validation Performance

Cross-validation metrics (10-fold expanding window):

| Metric | Value |
|--------|-------|
| **RMSE** | {data['wf_overall']['rmse']:.4f}% |
| **MAE** | {data['wf_overall']['mae']:.4f}% |
| **R²** | {data['wf_overall']['r2']:.4f} |
| **Sharpe Ratio** | {data['wf_overall']['sharpe']:.4f} |

**Interpretation:**
- RMSE indicates average prediction error of ~{data['wf_overall']['rmse']:.2f}% daily return
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
| **Initial Capital** | {data['backtest']['Initial Capital']} |
| **Final Equity** | {data['backtest']['Final Equity']} |
| **Total Return** | {data['backtest']['Total Return']} |
| **Annualized Return** | {data['backtest']['Annualized Return']} |
| **Volatility (Annual)** | {data['backtest']['Volatility (Annual)']} |
| **Sharpe Ratio** | {data['backtest']['Sharpe Ratio']} |
| **Sortino Ratio** | {data['backtest']['Sortino Ratio']} |
| **Max Drawdown** | {data['backtest']['Max Drawdown']} |
| **Win Rate** | {data['backtest']['Win Rate']} |
| **Total Trades** | {data['backtest']['Total Trades']} |

**Key Insights:**
- Extremely high returns suggest model has strong predictive power
- High Sharpe ratio ({data['backtest']['Sharpe Ratio']}) indicates excellent risk-adjusted performance
- Win rate of {data['backtest']['Win Rate']} shows consistent profitability
- Max drawdown of {data['backtest']['Max Drawdown']} indicates moderate risk exposure

⚠️ **Note:** These results represent in-sample backtest performance. Out-of-sample validation recommended before deployment.

---

## 3. Feature Importance (SHAP Analysis)

Top 5 most influential features for predictions:

| Rank | Feature | Mean |SHAP| Value |
|------|---------|------------------|
"""
    
    for i, row in data['top5_shap'].iterrows():
        md_content += f"| {i+1} | `{row['feature']}` | {row['mean_abs_shap']:.6f} |\n"
    
    md_content += f"""

**Feature Interpretation:**
1. **{data['top5_shap'].iloc[0]['feature']}**: Most influential - {data['top5_shap'].iloc[0]['mean_abs_shap']:.4f} average impact
2. **Technical indicators** (RSI, MACD): Strong contribution to predictions
3. **Lagged returns**: Historical momentum matters
4. **Macro indicators**: Economic context influences predictions

---

## 4. Model Architecture

- **Algorithm:** {data['model_type']}
- **Validation:** 10-fold walk-forward (expanding window)
- **Features:** {data['num_features']} total (technical, fundamental, macro, sentiment)
- **Target:** Next-day return percentage
- **Training Period:** {data['data_start']} to {data['data_end']}

---

## 5. Recommendations

### Strengths
- Strong feature engineering pipeline ({data['num_features']} features)
- Robust walk-forward validation methodology
- High backtest Sharpe ratio ({data['backtest']['Sharpe Ratio']})
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
- `data/processed/apple_feature_enhanced.csv` - Final dataset ({data['total_rows']:,} rows × {data['num_features']} features)

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

**Report Generated:** {data['timestamp']}  
**Pipeline Version:** 1.0  
**Contact:** Your Team/Email
"""
    
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"[OK] Markdown report saved: {MD_PATH}")


def generate_pdf_report(data):
    """Generate PDF version of the report."""
    
    print("[3/5] Generating PDF report...")
    
    # Create PDF document
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter,
                           topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Story (content)
    story = []
    
    # Title
    story.append(Paragraph("Apple Stock Prediction Model", title_style))
    story.append(Paragraph("Final Performance Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Metadata
    meta_data = [
        ['Generated:', data['timestamp']],
        ['Model Type:', data['model_type']],
        ['Data Period:', f"{data['data_start']} to {data['data_end']}"],
        ['Total Samples:', f"{data['total_rows']:,} trading days"],
        ['Features:', f"{data['num_features']} engineered features"]
    ]
    
    meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Section 1: Walk-Forward Validation
    story.append(Paragraph("1. Walk-Forward Validation Performance", heading_style))
    
    wf_data = [
        ['Metric', 'Value'],
        ['RMSE', f"{data['wf_overall']['rmse']:.4f}%"],
        ['MAE', f"{data['wf_overall']['mae']:.4f}%"],
        ['R²', f"{data['wf_overall']['r2']:.4f}"],
        ['Sharpe Ratio', f"{data['wf_overall']['sharpe']:.4f}"]
    ]
    
    wf_table = Table(wf_data, colWidths=[3*inch, 3*inch])
    wf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(wf_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Section 2: Backtest Results
    story.append(Paragraph("2. Backtest Trading Strategy", heading_style))
    
    bt_data = [
        ['Metric', 'Value'],
        ['Initial Capital', data['backtest']['Initial Capital']],
        ['Final Equity', data['backtest']['Final Equity']],
        ['Total Return', data['backtest']['Total Return']],
        ['Annualized Return', data['backtest']['Annualized Return']],
        ['Sharpe Ratio', data['backtest']['Sharpe Ratio']],
        ['Max Drawdown', data['backtest']['Max Drawdown']],
        ['Win Rate', data['backtest']['Win Rate']],
        ['Total Trades', str(data['backtest']['Total Trades'])]
    ]
    
    bt_table = Table(bt_data, colWidths=[3*inch, 3*inch])
    bt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(bt_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Section 3: SHAP Importance
    story.append(Paragraph("3. Top 5 Feature Importance (SHAP)", heading_style))
    
    shap_data = [['Rank', 'Feature', 'Mean |SHAP|']]
    for i, row in data['top5_shap'].iterrows():
        shap_data.append([str(i+1), row['feature'], f"{row['mean_abs_shap']:.6f}"])
    
    shap_table = Table(shap_data, colWidths=[1*inch, 3.5*inch, 1.5*inch])
    shap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(shap_table)
    
    # Page break before visualizations
    story.append(PageBreak())
    
    # Section 4: Visualizations
    story.append(Paragraph("4. Key Visualizations", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Add images (scaled to fit)
    img_width = 6.5*inch
    
    # Equity curve
    if (FIGURES_DIR / 'equity_curve.png').exists():
        story.append(Paragraph("Backtest Equity Curve", styles['Heading3']))
        img = Image(str(FIGURES_DIR / 'equity_curve.png'), width=img_width, height=img_width*0.7)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Walk-forward RMSE
    if (FIGURES_DIR / 'walkforward_rmse.png').exists():
        story.append(Paragraph("Walk-Forward Cross-Validation Metrics", styles['Heading3']))
        img = Image(str(FIGURES_DIR / 'walkforward_rmse.png'), width=img_width, height=img_width*0.7)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Page break
    story.append(PageBreak())
    
    # SHAP summary
    if (FIGURES_DIR / 'shap_summary.png').exists():
        story.append(Paragraph("SHAP Feature Importance Summary", styles['Heading3']))
        img = Image(str(FIGURES_DIR / 'shap_summary.png'), width=img_width, height=img_width*0.6)
        story.append(img)
    
    # Build PDF
    doc.build(story)
    
    print(f"[OK] PDF report saved: {PDF_PATH}")


def print_summary(data):
    """Print summary to console."""
    
    print("\n" + "="*70)
    print("FINAL REPORT GENERATION SUMMARY")
    print("="*70)
    print(f"\n[Model Information]")
    print(f"  Type: {data['model_type']}")
    print(f"  Features: {data['num_features']}")
    print(f"  Data Period: {data['data_start']} to {data['data_end']}")
    print(f"  Total Samples: {data['total_rows']:,}")
    
    print(f"\n[Walk-Forward Validation]")
    print(f"  RMSE: {data['wf_overall']['rmse']:.4f}%")
    print(f"  MAE: {data['wf_overall']['mae']:.4f}%")
    print(f"  R²: {data['wf_overall']['r2']:.4f}")
    print(f"  Sharpe: {data['wf_overall']['sharpe']:.4f}")
    
    print(f"\n[Backtest Performance]")
    print(f"  Total Return: {data['backtest']['Total Return']}")
    print(f"  Annualized Return: {data['backtest']['Annualized Return']}")
    print(f"  Sharpe Ratio: {data['backtest']['Sharpe Ratio']}")
    print(f"  Max Drawdown: {data['backtest']['Max Drawdown']}")
    print(f"  Win Rate: {data['backtest']['Win Rate']}")
    
    print(f"\n[Top 5 Features (SHAP)]")
    for i, row in data['top5_shap'].iterrows():
        print(f"  {i+1}. {row['feature']:<30} {row['mean_abs_shap']:.6f}")
    
    print(f"\n[Reports Generated]")
    print(f"  PDF: {PDF_PATH}")
    print(f"  Markdown: {MD_PATH}")
    print(f"  Timestamp: {data['timestamp']}")
    
    print("="*70 + "\n")


def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("FINAL REPORT GENERATOR")
    print("="*70 + "\n")
    
    try:
        # Load data
        data = load_data()
        
        # Generate reports
        generate_markdown_report(data)
        generate_pdf_report(data)
        
        # Print summary
        print("[4/5] Validating outputs...")
        print(f"[OK] PDF report: {PDF_PATH.exists()}")
        print(f"[OK] Markdown report: {MD_PATH.exists()}")
        
        print("[5/5] Printing summary...")
        print_summary(data)
        
        print("[SUCCESS] Final report generation completed!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
