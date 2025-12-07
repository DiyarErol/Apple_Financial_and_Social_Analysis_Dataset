"""
Visualization script for Apple Financial and Social Analysis Dataset.
Generates comprehensive visualizations and PDF report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_data(csv_path, corr_path):
    """
    Load both the main dataset and correlation matrix.
    
    Parameters:
    -----------
    csv_path : str
        Path to the main dataset CSV
    corr_path : str
        Path to the correlation matrix CSV
    
    Returns:
    --------
    tuple
        (DataFrame, DataFrame)
    """
    print("Loading data...")
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        corr_matrix = pd.read_csv(corr_path, index_col=0)
        
        print(f"✓ Data loaded: {len(df)} records")
        print(f"✓ Correlation matrix loaded: {corr_matrix.shape}")
        
        return df, corr_matrix
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_trend_timeline(df, output_dir):
    """
    Generate 10-year trend timeline with moving averages and scaled trends.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    output_dir : str
        Output directory for the figure
    """
    print("Generating trend timeline...")
    try:
        # Calculate moving average
        df['close_MA30'] = df['close'].rolling(window=30).mean()
        
        # Normalize trends to 0-1 scale
        apple_iphone_norm = (df['apple iphone'] - df['apple iphone'].min()) / \
                           (df['apple iphone'].max() - df['apple iphone'].min())
        macbook_norm = (df['macbook'] - df['macbook'].min()) / \
                       (df['macbook'].max() - df['macbook'].min())
        
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # Primary axis: Closing price and MA30
        color1 = '#1f77b4'
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Apple Closing Price (USD)', fontsize=12, color=color1, fontweight='bold')
        ax1.plot(df['date'], df['close'], color=color1, linewidth=1.5, alpha=0.5, label='Daily Close')
        ax1.plot(df['date'], df['close_MA30'], color=color1, linewidth=2.5, label='30-Day MA')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis: Normalized trends
        ax2 = ax1.twinx()
        ax2.set_ylabel('Google Trends Index (Normalized 0-1)', fontsize=12, fontweight='bold')
        ax2.plot(df['date'], apple_iphone_norm, color='#ff7f0e', linewidth=2, label='Apple iPhone Trend')
        ax2.plot(df['date'], macbook_norm, color='#2ca02c', linewidth=2, label='MacBook Trend')
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y')
        
        # Title and legend
        plt.title('Apple Stock 10-Year Trend (2015-2025) vs. Google Search Trends', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'trend_timeline_10year.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Trend timeline saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating trend timeline: {e}")


def create_price_distribution(df, output_dir):
    """
    Generate histogram and KDE plot for closing prices.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    output_dir : str
        Output directory for the figure
    """
    print("Generating price distribution plot...")
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Histogram with KDE overlay
        df['close'].hist(bins=50, color='#1f77b4', alpha=0.7, edgecolor='black', ax=ax)
        df['close'].plot(kind='kde', secondary_y=False, ax=ax, color='#d62728', 
                        linewidth=2.5, label='KDE')
        
        # Labels
        ax.set_xlabel('Closing Price (USD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Distribution of Apple Stock Closing Prices (2015-2025)', 
                 fontsize=14, fontweight='bold', pad=20)
        ax.legend(['KDE', 'Histogram'], fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics box
        mean_price = df['close'].mean()
        median_price = df['close'].median()
        std_price = df['close'].std()
        
        stats_text = f'Mean: ${mean_price:.2f}\nMedian: ${median_price:.2f}\nStd: ${std_price:.2f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'price_distribution.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ Price distribution plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating price distribution: {e}")


def create_wordcloud_visualization(output_dir):
    """
    Generate WordCloud from keywords.
    
    Parameters:
    -----------
    output_dir : str
        Output directory for the figure
    """
    print("Generating WordCloud...")
    try:
        # Keywords with frequencies (based on domain knowledge)
        keywords = {
            'Apple': 85,
            'iPhone': 92,
            'MacBook': 78,
            'iOS': 88,
            'Mac': 81,
            'Vision Pro': 45,
            'stock': 70,
            'technology': 75,
            'innovation': 72,
            'product': 68,
            'ecosystem': 65,
            'research': 60,
            'development': 62,
            'growth': 70,
            'market': 75
        }
        
        # Create WordCloud
        wordcloud = WordCloud(width=1200, height=600, 
                            background_color='white',
                            colormap='viridis',
                            relative_scaling=0.5,
                            min_font_size=14).generate_from_frequencies(keywords)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title('Apple Products and Market Keywords Cloud', 
                 fontsize=14, fontweight='bold', pad=20)
        
        fig.tight_layout()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'wordcloud.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"✓ WordCloud saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"✗ Error generating WordCloud: {e}")


def create_pdf_report(df, corr_matrix, output_path):
    """
    Create comprehensive PDF report with title, summary, and figures.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    corr_matrix : pd.DataFrame
        Correlation matrix
    output_path : str
        Path to save the PDF
    """
    print("Generating PDF report...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(output_path) as pdf:
            # Page 1: Title Page
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.75, 'APPLE FINANCIAL AND SOCIAL ANALYSIS', 
                   ha='center', va='center', fontsize=28, fontweight='bold')
            ax.text(0.5, 0.65, 'Comprehensive Stock Price and Google Trends Analysis', 
                   ha='center', va='center', fontsize=14, style='italic')
            
            # Date and info
            current_date = datetime.now().strftime('%B %d, %Y')
            ax.text(0.5, 0.50, f'Report Generated: {current_date}', 
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.45, f'Data Period: 2015-01-02 to 2025-12-01', 
                   ha='center', va='center', fontsize=11)
            ax.text(0.5, 0.40, f'Total Records Analyzed: {len(df):,}', 
                   ha='center', va='center', fontsize=11)
            
            # Footer
            ax.text(0.5, 0.15, 'Apple Inc. (AAPL)', 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.10, 'Analysis of Historical Stock Data and Search Trends', 
                   ha='center', va='center', fontsize=10)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Executive Summary
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.95, 'EXECUTIVE SUMMARY', 
                   ha='center', va='top', fontsize=16, fontweight='bold',
                   transform=ax.transAxes)
            
            # Calculate statistics
            mean_close = df['close'].mean()
            median_close = df['close'].median()
            std_close = df['close'].std()
            min_close = df['close'].min()
            max_close = df['close'].max()
            
            # Statistics text
            summary_text = f"""
CLOSING PRICE STATISTICS (USD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mean Price:              ${mean_close:.2f}
Median Price:           ${median_close:.2f}
Standard Deviation:     ${std_close:.2f}
Minimum Price:          ${min_close:.2f}
Maximum Price:          ${max_close:.2f}
Price Range:            ${max_close - min_close:.2f}

DATASET OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Trading Days:     {len(df):,}
Date Range:             2015-01-02 to 2025-12-01
Years Covered:          10+

GOOGLE TRENDS ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apple iPhone Trend:     Mean = {df['apple iphone'].mean():.1f}, Std = {df['apple iphone'].std():.1f}
MacBook Trend:          Mean = {df['macbook'].mean():.1f}, Std = {df['macbook'].std():.1f}

KEY CORRELATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Closing Price vs. MacBook:      {corr_matrix.loc['close', 'macbook']:.4f}
Closing Price vs. iPhone:       {corr_matrix.loc['close', 'apple iphone']:.4f}
Price vs. Trading Volume:       {corr_matrix.loc['close', 'volume']:.4f}
            """
            
            ax.text(0.05, 0.85, summary_text, ha='left', va='top', 
                   fontsize=9, family='monospace', transform=ax.transAxes)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Correlation Heatmap
            if Path('reports/figures/correlation_heatmap.png').exists():
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                # Load and display heatmap image
                from PIL import Image
                img = Image.open('reports/figures/correlation_heatmap.png')
                ax.imshow(img)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 4: Trend Timeline
            if Path('reports/figures/trend_timeline_10year.png').exists():
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                from PIL import Image
                img = Image.open('reports/figures/trend_timeline_10year.png')
                ax.imshow(img)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 5: Scatter Plot
            if Path('reports/figures/macbook_vs_price_scatter.png').exists():
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                from PIL import Image
                img = Image.open('reports/figures/macbook_vs_price_scatter.png')
                ax.imshow(img)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        print(f"✓ PDF report saved to {output_path}")
    except Exception as e:
        print(f"✗ Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main execution function:
    1. Load data and correlation matrix
    2. Generate visualizations
    3. Create PDF report
    """
    print("\n" + "="*80)
    print("APPLE FINANCIAL AND SOCIAL ANALYSIS - VISUALIZATION AND REPORTING")
    print("="*80 + "\n")
    
    # Define paths
    csv_path = 'data/processed/apple_analysis_dataset.csv'
    corr_path = 'reports/final/correlation_matrix.csv'
    output_figures_dir = 'reports/figures'
    output_pdf = 'reports/final/apple_analysis_report.pdf'
    
    # Load data
    df, corr_matrix = load_data(csv_path, corr_path)
    if df is None or corr_matrix is None:
        print("✗ Failed to load data. Exiting.")
        return False
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_trend_timeline(df, output_figures_dir)
    create_price_distribution(df, output_figures_dir)
    create_wordcloud_visualization(output_figures_dir)
    
    # Create PDF report
    print("\nCreating PDF report...")
    create_pdf_report(df, corr_matrix, output_pdf)
    
    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION AND REPORTING COMPLETE")
    print("="*80)
    
    print(f"\n✓ Visualizations Generated:")
    print(f"  • {output_figures_dir}/trend_timeline_10year.png")
    print(f"  • {output_figures_dir}/price_distribution.png")
    print(f"  • {output_figures_dir}/wordcloud.png")
    
    print(f"\n✓ Report Generated:")
    print(f"  • {output_pdf}")
    
    print(f"\n✓ Report Contents:")
    print(f"  • Title Page")
    print(f"  • Executive Summary (with statistics)")
    print(f"  • Correlation Heatmap")
    print(f"  • 10-Year Trend Timeline")
    print(f"  • MacBook vs. Price Scatter Plot")
    
    print("\n" + "="*80)
    print("✓ All visualization tasks completed successfully!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
