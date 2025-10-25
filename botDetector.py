import praw
from dotenv import load_dotenv
import os
import numpy as np
import time
from datetime import datetime
from collections import Counter
import math

load_dotenv()

# --- Reddit API setup ---
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="bot detection unwrapathon"
)

from prawcore.exceptions import NotFound, Forbidden, PrawcoreException

def fetch_user_data_safe(username, limit=50):
    """Fetch user data; return None if user doesn't exist, suspended, or private."""
    try:
        user = reddit.redditor(username)
        created = user.created_utc  # triggers actual fetch
        user_data = {
            "username": username,
            "created_utc": created,
            "link_karma": user.link_karma,
            "comment_karma": user.comment_karma,
            "posts": [],
            "comments": []
        }

        for post in user.submissions.new(limit=limit):
            user_data["posts"].append({
                "created_utc": post.created_utc,
                "subreddit": str(post.subreddit),
                "score": post.score
            })
        for com in user.comments.new(limit=limit):
            user_data["comments"].append({
                "created_utc": com.created_utc,
                "subreddit": str(com.subreddit),
                "score": com.score
            })
        return user_data

    except (NotFound, Forbidden):
        # user doesn't exist or suspended/private
        return None
    except PrawcoreException as e:
        print(f"[WARN] Could not fetch {username}: {e}")
        return None

from collections import Counter
import re
from difflib import SequenceMatcher

import re

def clean_text(text: str) -> str:
    """Basic text normalization for Reddit posts/comments."""
    # Lowercase everything
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_duplicate_content_ratio(items):
    """Compute ratio of near-duplicate comments/titles (0–1)."""
    texts = [clean_text(i) for i in items if isinstance(i, str) and i.strip()]
    if len(texts) < 3:
        return 0.0
    
    sim_scores = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
            if sim > 0.8:  # consider “Thank you” and “Thanks!” same
                sim_scores.append(1)
            else:
                sim_scores.append(0)
    return sum(sim_scores) / max(len(sim_scores), 1)


def compute_features(user_data):
    """Compute normalized 0–1 heuristic features for bot likelihood."""
    now = time.time()
    age_days = (now - user_data["created_utc"]) / (60 * 60 * 24)
    posts = user_data["posts"]
    comments = user_data["comments"]

    # --- 1. Account age ---
    # Younger = more bot-like, with exponential decay over ~6 months
    age_score = np.exp(-age_days / 180)  # 1.0 new, ~0.03 at 1 year

    # --- 2. Comment-to-post ratio ---
    c_to_p = len(comments) / (len(posts) + 1e-6)
    # Extremely low comment activity is bot-like
    comment_to_post_score = 1 - min(1.0, c_to_p) if c_to_p < 1 else 0.0

    # --- 3. Subreddit diversity ---
    subreddits = [p["subreddit"] for p in posts] + [c["subreddit"] for c in comments]
    subreddit_count = len(set(subreddits))
    # Fewer subreddits = more bot-like
    subreddit_diversity_score = 1 - min(subreddit_count / 10, 1.0)  # 1 subreddit → 0.9, 10+ → 0.0

    # --- 4. Activity spikes ---
    times = sorted([x["created_utc"] for x in posts + comments])
    activity_spike_score = 0.0
    if len(times) > 5:
        deltas = np.diff(times)
        mean_gap = np.mean(deltas)
        if np.any((deltas[:-1] > mean_gap * 5) & (deltas[1:] < mean_gap / 5)):
            activity_spike_score = 1.0

    # --- 5. Post-to-karma ratio ---
    total_posts = max(len(posts), 1)
    total_karma = user_data["link_karma"] + user_data["comment_karma"]
    post_to_karma_ratio = total_posts / (total_karma + 1e-6)
    # High ratio = suspicious
    subreddit_diversity_score = 1 - min(subreddit_count / 5, 1.0)

    # --- 6. Duplicate content ---
    comment_dupe_ratio = get_duplicate_content_ratio(comments)
    post_dupe_ratio = get_duplicate_content_ratio(posts)
    duplicate_content_ratio = max(comment_dupe_ratio, post_dupe_ratio)
    duplicate_content_score = min(duplicate_content_ratio * 5, 1.0)

    # --- 7. Posting frequency ---
    posts_per_day = (len(posts) + len(comments)) / min(age_days, 30)
    # 0 = inactive, 1 = posting a ton
    posts_per_day_score = min(posts_per_day / 20, 1.0)

    return {
        "age_score": age_score,
        "comment_to_post_score": comment_to_post_score,
        "subreddit_diversity_score": subreddit_diversity_score,
        "activity_spike_score": activity_spike_score,
        "post_to_karma_score": post_to_karma_ratio,
        "duplicate_content_score": duplicate_content_score,
        "posts_per_day_score": posts_per_day_score,
    }


def age_penalty(age_days):
    return 0.25 * math.exp(-age_days / 90)


def subreddit_diversity_penalty(subreddit_count):
    return 0.3 * math.exp(-subreddit_count / 5)

# --- Heuristic scoring ---
def compute_bot_score(features):

    #weights
    AGE_WEIGHT = 0.25
    SR_DIVERSITY = 0.25
    ACTIVITY_SPIKE = 0.1
    PK_RATIO = 0.25
    POST_PER_DAY = 0.15



    score = 0.0

     # Age-based decay
    score += (age_penalty(features["age_score"]) * AGE_WEIGHT)

    # Subreddit diversity penalty
    score += (subreddit_diversity_penalty(features["subreddit_diversity_score"]) * SR_DIVERSITY)


    # 4. Activity spikes → +0.15
    if features["activity_spike_score"]:
        score += ACTIVITY_SPIKE

    # 5. Post-to-karma ratio unusually high → +0.15
    if features["post_to_karma_score"] > 0.1:  # tweak threshold
        score += PK_RATIO

    # (Optional)
    # 6. Posting rate > X per day → +0.25
    if features["posts_per_day_score"] > 10:
         score += POST_PER_DAY

    # 7. GPT-based text analysis
    # score += 0.2 * features["gpt_score"]

    return min(1.0, score)

def analyze_user(username):
    data = fetch_user_data_safe(username)
    features = compute_features(data)
    bot_score = compute_bot_score(features)
    return {
        "username": username,
        "bot_score": bot_score,
        "features": features
    }

def generate_bot_score(username):
    data = fetch_user_data_safe(username)
    if data is None:
        return None
    features = compute_features(data)
    return compute_bot_score(features)

if __name__ == "__main__":
    test_user = input("Enter Reddit username to analyze: ")
    result = analyze_user(test_user)
    print("\n--- Bot Score Report ---")
    print(f"Username: {result['username']}")
    print(f"Bot Score: {result['bot_score']:.2f}")
    print("Feature breakdown:")
    for k, v in result["features"].items():
        print(f"  {k}: {v}")