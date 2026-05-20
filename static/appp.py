"""
analyze_social_media_pipeline.py
- Fetches tweets and YouTube comments for a keyword
- Preprocesses text, runs VADER sentiment
- Extracts hashtags and top keywords (TF-IDF)
- Trains a simple classifier to predict 'post_success_label'
Notes: Provide API keys in .env or set as environment vars.
"""

import os
import re
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

import pandas as pd
import numpy as np

# NLP & ML
import nltk
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# API clients
import tweepy
from googleapiclient.discovery import build

# Initialize environment
load_dotenv()
TWITTER_BEARER = os.getenv("TWITTER_BEARER", "")
YT_API_KEY = os.getenv("YT_API_KEY", "")

# --- Ensure NLTK data ---
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------------------
# 1. Data fetchers
# ---------------------------
def fetch_tweets(query, max_results=200):
    """Fetch recent tweets using Twitter v2 tweepy.Client (requires bearer token)."""
    if not TWITTER_BEARER:
        raise RuntimeError("Set TWITTER_BEARER in environment or .env")
    client = tweepy.Client(bearer_token=TWITTER_BEARER, wait_on_rate_limit=True)
    tweets = []
    # Use pagination
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query + " -is:retweet lang:en",
        tweet_fields=["created_at","public_metrics","text","author_id"],
        max_results=100,
    )
    count = 0
    for page in paginator:
        if page.data is None:
            continue
        for t in page.data:
            tweets.append({
                "id": t.id,
                "platform": "twitter",
                "text": t.text,
                "timestamp": t.created_at.isoformat(),
                "likes": t.public_metrics.get("like_count",0),
                "shares": t.public_metrics.get("retweet_count",0)
            })
            count += 1
            if count >= max_results:
                return pd.DataFrame(tweets)
        # avoid overloading
        time.sleep(1)
    return pd.DataFrame(tweets)

def fetch_youtube_comments_for_video(video_id, max_comments=200):
    """Fetch comments for a single YouTube video"""
    if not YT_API_KEY:
        raise RuntimeError("Set YT_API_KEY in environment or .env")
    youtube = build("youtube", "v3", developerKey=YT_API_KEY)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "id": item["id"],
                "platform": "youtube",
                "text": s.get("textDisplay",""),
                "timestamp": s.get("publishedAt"),
                "likes": s.get("likeCount",0),
                "shares": 0
            })
            if len(comments) >= max_comments:
                break
        request = youtube.commentThreads().list_next(request, response)
    return pd.DataFrame(comments)

def search_youtube_videos(query, max_results=5):
    if not YT_API_KEY:
        raise RuntimeError("Set YT_API_KEY in environment or .env")
    youtube = build("youtube", "v3", developerKey=YT_API_KEY)
    resp = youtube.search().list(q=query, part="snippet", type="video", maxResults=max_results).execute()
    return [item["id"]["videoId"] for item in resp.get("items", [])]

# Placeholder Instagram fetcher: implement using official Graph API or third-party scraping as permitted.
def fetch_instagram_placeholder(query, max_results=0):
    return pd.DataFrame([])

# ---------------------------
# 2. Preprocessing & features
# ---------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s#]", " ", text)   # keep hashtags
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_hashtags(text):
    return [tok for tok in text.split() if tok.startswith("#")]

def compute_vader_sentiment(text, analyzer):
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

# ---------------------------
# 3. Pipeline
# ---------------------------
def run_pipeline(keyword, tweet_count=400, yt_videos=3, yt_comments_per_video=150):
    analyzer = SentimentIntensityAnalyzer()

    # 1) Fetch data
    print("Fetching tweets...")
    df_tweets = fetch_tweets(keyword, max_results=tweet_count)
    print(f"Fetched {len(df_tweets)} tweets.")

    print("Searching YouTube videos...")
    vids = search_youtube_videos(keyword, max_results=yt_videos)
    df_yt = []
    for v in vids:
        print(" Fetching comments for video:", v)
        dfc = fetch_youtube_comments_for_video(v, max_comments=yt_comments_per_video)
        df_yt.append(dfc)
    df_yt = pd.concat(df_yt, ignore_index=True) if df_yt else pd.DataFrame([])

    # combine
    df = pd.concat([df_tweets, df_yt], ignore_index=True, sort=False).fillna("")
    if df.empty:
        raise RuntimeError("No data fetched. Check API keys or query and try again.")

    # 2) Preprocess
    df["clean_text"] = df["text"].apply(preprocess_text)
    df["hashtags"] = df["clean_text"].apply(extract_hashtags)
    df["vader_compound"] = df["clean_text"].apply(lambda t: compute_vader_sentiment(t, analyzer))
    df["engagement"] = df[["likes","shares"]].sum(axis=1)

    # 3) Labeling heuristic for demonstration:
    # define post_success_label = 1 if (vader_compound > 0.3 and engagement >= median(engagement))
    med_eng = max(1, int(df["engagement"].median()))
    df["post_success_label"] = ((df["vader_compound"] > 0.3) & (df["engagement"] >= med_eng)).astype(int)

    # 4) TF-IDF features
    vectorizer = CountVectorizer(stop_words="english", max_features=3000)
    X_counts = vectorizer.fit_transform(df["clean_text"])
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_counts)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    # 5) Features and model
    features = np.hstack([df[["vader_compound","engagement"]].values, X_reduced])
    y = df["post_success_label"].values

    # small train/test split (if labels are imbalanced, consider resampling)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Model accuracy:", round(accuracy_score(y_test, y_pred),4))
    print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Top hashtags
    all_hashtags = sum(df["hashtags"].tolist(), [])
    top_hashtags = pd.Series(all_hashtags).value_counts().head(15)

    # return artifacts
    artifacts = {
        "df": df,
        "vectorizer": vectorizer,
        "tfidf": tfidf,
        "svd": svd,
        "model": model,
        "top_hashtags": top_hashtags
    }
    return artifacts

# ---------------------------
# 4. Main runner
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run social media analysis pipeline")
    parser.add_argument("--query", "-q", required=True, help="Search keyword (product or brand)")
    parser.add_argument("--tweets", type=int, default=300, help="Number of tweets to fetch")
    parser.add_argument("--yt_videos", type=int, default=2, help="Number of YouTube videos to scrape comments from")
    parser.add_argument("--yt_comments", type=int, default=150, help="Number of comments per video")
    args = parser.parse_args()

    arts = run_pipeline(args.query, tweet_count=args.tweets, yt_videos=args.yt_videos, yt_comments_per_video=args.yt_comments)
    print("Top hashtags:\n", arts["top_hashtags"].to_string())
    # Save model artifact
    import pickle
    with open("sms_pipeline_artifacts.pkl", "wb") as f:
        pickle.dump({
            "model": arts["model"],
            "vectorizer": arts["vectorizer"],
            "tfidf": arts["tfidf"],
            "svd": arts["svd"]
        }, f)
    print("Saved artifacts to sms_pipeline_artifacts.pkl")
