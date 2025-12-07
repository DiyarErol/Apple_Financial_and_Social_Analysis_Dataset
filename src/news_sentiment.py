"""
News Sentiment Integration Script
Apple Financial and Social Analysis Dataset

This script fetches daily Apple news via NewsAPI, scores sentiment with FinBERT,
creates daily aggregates, merges them into the dataset, and plots a timeline.

Requirements:
- Environment variable: NEWSAPI_KEY
- Packages: requests, pandas, numpy, matplotlib, transformers, torch (for FinBERT)
- Respects a 100-requests-per-run cap to align with NewsAPI daily limits.
"""

import os
import math
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FINBERT_MODEL = "ProsusAI/finbert"
MAX_REQUESTS_PER_RUN = 100  # safety cap for NewsAPI daily limit


# ============================================================================
# 1. Data Loading
# ============================================================================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("NEWS SENTIMENT PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("=" * 80)
    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ============================================================================
# 2. News Fetching
# ============================================================================

def fetch_news_for_date(date: pd.Timestamp, api_key: str, query: str = "Apple Inc"):
    """Fetch news titles for a specific date from NewsAPI."""
    params = {
        "q": query,
        "from": date.strftime("%Y-%m-%d"),
        "to": date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠ {date.date()}: NewsAPI returned {resp.status_code} ({resp.text[:120]})")
            return []
        data = resp.json()
        articles = data.get("articles", [])
        titles = [a.get("title") for a in articles if a.get("title")]
        return titles
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"  ⚠ {date.date()}: Error fetching news ({exc})")
        return []


def fetch_news_range(dates: pd.Series, api_key: str) -> dict:
    """Fetch news per date up to MAX_REQUESTS_PER_RUN."""
    results = {}
    req_count = 0

    sorted_dates = sorted(pd.to_datetime(dates.unique()))
    for dt in sorted_dates:
        if req_count >= MAX_REQUESTS_PER_RUN:
            print(f"\n⚠ Request cap reached ({MAX_REQUESTS_PER_RUN}). Remaining dates left as NaN.")
            break

        titles = fetch_news_for_date(dt, api_key)
        results[dt.normalize()] = titles
        req_count += 1
        time.sleep(0.2)  # small pause to be kind to the API

    print(f"\n✓ News fetched for {len(results)} days (cap {MAX_REQUESTS_PER_RUN})")
    return results


# ============================================================================
# 3. Sentiment with FinBERT
# ============================================================================

def load_finbert_pipeline():
    print("\n[3/7] Loading FinBERT model...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("transformers is required for FinBERT sentiment analysis.") from exc

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    print("✓ FinBERT loaded")
    return nlp


def score_texts_finbert(texts, nlp):
    """Return list of sentiment scores in [-1, +1] using FinBERT (pos - neg)."""
    scores = []
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        outputs = nlp(text[:512])  # truncate long titles to stay efficient
        # outputs: list of list of dicts
        probs = {item['label'].lower(): item['score'] for item in outputs[0]}
        pos = probs.get('positive', 0.0)
        neg = probs.get('negative', 0.0)
        score = pos - neg
        scores.append(score)
    return scores


def aggregate_daily_sentiment(titles_by_date: dict, nlp) -> pd.DataFrame:
    records = []
    for dt, titles in titles_by_date.items():
        if not titles:
            records.append({
                'date': dt,
                'news_sentiment_score': np.nan,
                'news_count': 0,
                'positive_news_ratio': np.nan,
                'negative_news_ratio': np.nan,
            })
            continue

        scores = score_texts_finbert(titles, nlp)
        if not scores:
            records.append({
                'date': dt,
                'news_sentiment_score': np.nan,
                'news_count': len(titles),
                'positive_news_ratio': np.nan,
                'negative_news_ratio': np.nan,
            })
            continue

        mean_score = float(np.mean(scores))
        pos_count = sum(1 for s in scores if s > 0)
        neg_count = sum(1 for s in scores if s < 0)
        total = len(scores)
        records.append({
            'date': dt,
            'news_sentiment_score': mean_score,
            'news_count': total,
            'positive_news_ratio': pos_count / total if total else np.nan,
            'negative_news_ratio': neg_count / total if total else np.nan,
        })

    df_sent = pd.DataFrame(records)
    return df_sent


# ============================================================================
# 4. Merge with dataset
# ============================================================================

def merge_sentiment(df: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/7] Merging sentiment with main dataset...")
    merged = df.merge(df_sent, on='date', how='left')
    print(f"✓ Merged: {len(merged):,} rows, {len(merged.columns)} columns")
    return merged


# ============================================================================
# 5. Plotting
# ============================================================================

def plot_sentiment_timeline(df: pd.DataFrame, output_path: Path):
    print("\n[6/7] Plotting sentiment timeline...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    color_price = '#1f77b4'
    ax1.plot(df['date'], df['close'], color=color_price, label='Close Price', linewidth=1.5)
    ax1.set_ylabel('Close Price ($)', color=color_price, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_price)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_sent = '#ff7f0e'
    ax2.plot(df['date'], df['news_sentiment_score'], color=color_sent, label='News Sentiment', linewidth=1.0, alpha=0.7)
    ax2.set_ylabel('News Sentiment (FinBERT)', color=color_sent, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_sent)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.set_ylim(-1.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    plt.title('Apple Price vs. News Sentiment (FinBERT)', fontsize=14, fontweight='bold')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Timeline saved: {output_path}")


# ============================================================================
# 6. Main
# ============================================================================

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data' / 'processed'
    reports_dir = script_dir / 'reports' / 'figures'

    input_csv = data_dir / 'apple_with_events.csv'
    output_csv = data_dir / 'apple_with_sentiment.csv'
    output_fig = reports_dir / 'news_sentiment_timeline.png'

    # Load dataset
    print("\n" + "=" * 80)
    print("NEWS SENTIMENT PIPELINE")
    print("Apple Financial and Social Analysis Dataset")
    print("=" * 80)
    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Check API key
    print("\n[2/7] Checking NEWSAPI_KEY...")
    api_key = os.environ.get('NEWSAPI_KEY')
    if not api_key:
        print("⚠ NEWSAPI_KEY environment variable not found.")
        print("  Creating placeholder sentiment columns (NaN values)...\n")
        
        # Create placeholder sentiment columns
        df['news_sentiment_score'] = np.nan
        df['news_count'] = 0
        df['positive_news_ratio'] = np.nan
        df['negative_news_ratio'] = np.nan
        
        # Save
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[3/7] Saved placeholder dataset: {output_csv}")
        print(f"✓ Columns added: news_sentiment_score, news_count, positive_news_ratio, negative_news_ratio")
        
        print("\n" + "=" * 80)
        print("NEWS SENTIMENT PIPELINE COMPLETE (PLACEHOLDER MODE)")
        print("=" * 80)
        print(f"\n✓ Output CSV: {output_csv}")
        print("ℹ To enable actual sentiment analysis, set NEWSAPI_KEY environment variable")
        print("  Get free API key from: https://newsapi.org")
        return

    # Fetch news (capped at 100 requests per run)
    print("✓ NEWSAPI_KEY found")
    print("\n[2/7] Fetching news from NewsAPI (capped at 100 requests)...")
    titles_by_date = fetch_news_range(df['date'], api_key)

    # Load FinBERT and score
    nlp = load_finbert_pipeline()
    print("\n[4/7] Scoring daily sentiment...")
    df_sent = aggregate_daily_sentiment(titles_by_date, nlp)

    # Merge
    df_merged = merge_sentiment(df, df_sent)

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    print(f"\n[6/7] Saved merged dataset: {output_csv} ({len(df_merged):,} rows)")

    # Plot
    plot_sentiment_timeline(df_merged, output_fig)

    # Summary
    print("\n" + "=" * 80)
    print("NEWS SENTIMENT PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n✓ Output CSV: {output_csv}")
    print(f"✓ Output Figure: {output_fig}")
    print(f"✓ News days fetched: {len(titles_by_date)} (cap {MAX_REQUESTS_PER_RUN})")


if __name__ == '__main__':
    main()
