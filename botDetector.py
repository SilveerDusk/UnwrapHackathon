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

def fetch_user_data(username, limit=50):
    """Fetch user metadata, latest posts & comments."""
    user = reddit.redditor(username)

    user_data = {
        "username": username,
        "created_utc": user.created_utc,
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

from collections import Counter
import re

def get_duplicate_content_ratio(items):
    """Compute the fraction of duplicate texts in a list of posts/comments.
    Handles strings, dicts, or PRAW objects gracefully.
    """
    if not items:
        return 0.0

    def extract_text(i):
        # Handle raw string
        if isinstance(i, str):
            return i
        # Handle dict with a 'body' or 'title' field
        elif isinstance(i, dict):
            return i.get("body") or i.get("title") or ""
        # Handle PRAW Comment or Submission
        elif hasattr(i, "body"):
            return i.body
        elif hasattr(i, "title"):
            return i.title
        else:
            return ""

    def clean_text(t):
        t = t.lower()
        t = re.sub(r"http\S+", "", t)  # remove URLs
        return t.strip()

    # Extract + clean all text items
    cleaned = [clean_text(extract_text(i)) for i in items if extract_text(i).strip()]
    if not cleaned:
        return 0.0

    counts = Counter(cleaned)
    duplicates = sum(c for c in counts.values() if c > 1)
    ratio = duplicates / len(cleaned)

    return round(ratio, 3)


def compute_features(user_data):
    """Compute heuristic features from user_data."""
    now = time.time()
    age_days = (now - user_data["created_utc"]) / (60 * 60 * 24)
    posts = user_data["posts"]
    comments = user_data["comments"]

    #comments = [c.body for c in user.comments.new(limit=50)]
    #posts = [s.title for s in user.submissions.new(limit=30)]

    # Comment-to-post ratio
    c_to_p = len(comments) / (len(posts) + 1e-6)

    # Subreddit diversity
    subreddits = [p["subreddit"] for p in posts] + [c["subreddit"] for c in comments]
    subreddit_count = len(set(subreddits))

    # --- Activity spikes ---
    # Sort all timestamps
    times = sorted([x["created_utc"] for x in posts + comments])
    activity_spike = False
    if len(times) > 5:
        deltas = np.diff(times)
        mean_gap = np.mean(deltas)
        # If there exists a gap 5× longer than average followed by dense activity → spike
        if np.any((deltas[:-1] > mean_gap * 5) & (deltas[1:] < mean_gap / 5)):
            activity_spike = True

    # --- Post-to-karma ratio ---
    total_posts = max(len(posts), 1)
    total_karma = user_data["link_karma"] + user_data["comment_karma"]
    post_to_karma_ratio = total_posts / (total_karma + 1e-6)

    comment_dupe_ratio = get_duplicate_content_ratio(comments)
    post_dupe_ratio = get_duplicate_content_ratio(posts)
    duplicate_content_ratio = max(comment_dupe_ratio, post_dupe_ratio)

    # (Optional)
    # --- Posting rate ---
    # posts_per_day = (len(posts) + len(comments)) / min(age_days, 30)
    # --- GPT text analysis ---
    # gpt_score = get_gpt_score(user_data)

    return {
        "age_days": age_days,
        "comment_to_post_ratio": c_to_p,
        "subreddit_count": subreddit_count,
        "activity_spike": activity_spike,
        "post_to_karma_ratio": post_to_karma_ratio,
        "duplicate_content_ratio": duplicate_content_ratio
        # "posts_per_day": posts_per_day,
        # "gpt_score": gpt_score
    }

def age_penalty(age_days):
    return 0.25 * math.exp(-age_days / 90)


def subreddit_diversity_penalty(subreddit_count):
    return 0.3 * math.exp(-subreddit_count / 5)

# --- Heuristic scoring ---
def compute_bot_score(features):
    score = 0.0

     # Age-based decay
    score += age_penalty(features["age_days"])

    # Subreddit diversity penalty
    score += subreddit_diversity_penalty(features["subreddit_count"])


    # 4. Activity spikes → +0.15
    if features["activity_spike"]:
        score += 0.15

    # 5. Post-to-karma ratio unusually high → +0.15
    if features["post_to_karma_ratio"] > 0.1:  # tweak threshold
        score += 0.15

    # (Optional)
    # 6. Posting rate > X per day → +0.25
    # if features["posts_per_day"] > 5:
    #     score += 0.25

    # 7. GPT-based text analysis
    # score += 0.2 * features["gpt_score"]

    return min(1.0, score)

def analyze_user(username):
    data = fetch_user_data(username)
    features = compute_features(data)
    bot_score = compute_bot_score(features)
    return {
        "username": username,
        "bot_score": bot_score,
        "features": features
    }


if __name__ == "__main__":
    test_user = input("Enter Reddit username to analyze: ")
    result = analyze_user(test_user)
    print("\n--- Bot Score Report ---")
    print(f"Username: {result['username']}")
    print(f"Bot Score: {result['bot_score']:.2f}")
    print("Feature breakdown:")
    for k, v in result["features"].items():
        print(f"  {k}: {v}")