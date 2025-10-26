import joblib
import pandas as pd
import praw
#from botGroundBuilder import analyze_user
import os 
from collections import Counter
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv

import re

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import time
import numpy as np


load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="bot detection unwrapathon"
)

subreddit_name = "BotBouncer"
subreddit = reddit.subreddit(subreddit_name)

def generate_user_table(usernames, parquet_file="users.parquet") -> None:
    results = []

    # check if user exists
    if os.path.exists(parquet_file):
        existing_df = pd.read_parquet(parquet_file, engine="fastparquet")
        existing_usernames = set(existing_df['username'])
    else:
        existing_df = pd.DataFrame()
        existing_usernames = set()

    for uid in usernames:
        if uid in existing_usernames:
            continue  # skip duplicates

        usr_feats = analyze_user(uid)
        print(uid)
        print(usr_feats)

        if usr_feats is not None:
            usr_feats['username'] = uid
            results.append(usr_feats)
    
    print(f"Results: {results}")

    if not results:
        return  # nothing new to add

    # Create DataFrame for new users
    new_users_df = pd.DataFrame(results)

    # Concatenate with existing data if any
    if not existing_df.empty:
        user_df = pd.concat([existing_df, new_users_df], ignore_index=True)
    else:
        user_df = new_users_df

    # Save back to Parquet
    user_df.to_parquet(parquet_file, index=False)




def classify_bots(usernames, parquet_file="user_scores.parquet"):

    rf_pipeline = joblib.load("bot_detector.pkl")

    # check if user exists
    if os.path.exists(parquet_file):
        user_scores_df = pd.read_parquet(parquet_file, engine="fastparquet")
    else:
        user_scores_df = pd.DataFrame()

    user_df = pd.read_parquet("users.parquet", engine="fastparquet")
    

    results = []

    for uid in usernames:
        # Make sure columns match training
        user_row = user_df[user_df['username']==uid]

        if user_row.empty:
            continue  # skip if user not found

        user_row = user_row.drop(columns=['username'])

        # Get probability of being a bot
        bot_prob = rf_pipeline.predict_proba(user_row)[:, 1][0]

        # Apply your threshold
        threshold = 0.6
        is_bot = int(bot_prob >= threshold)

        results.append({
            'username': uid,
            'score': bot_prob,
        })
    
    new_scores = pd.DataFrame(results)

    user_scores_df = pd.concat([user_scores_df, new_scores], ignore_index=True)

    # Save to Parquet
    user_scores_df.to_parquet(parquet_file, index=False)
        #print(f"Predicted bot probability: {bot_prob:.3f}")
        #print(f"Predicted label: {'Bot' if is_bot else 'Human'}")


def fetch_user_data_safe(username, limit=50):
    try:
        user = reddit.redditor(username)

        # If user doesn't exist or is suspended, they won't have created_utc
        if not hasattr(user, "created_utc"):
            print(f"⚠️ Skipping user '{username}' (no created_utc — likely suspended or deleted)")
            return None

        created = user.created_utc
        user_data = {
            "username": username,
            "created_utc": created,
            "posts": [],
            "comments": []
        }

        # Try fetching recent posts and comments safely
        try:
            for post in user.submissions.new(limit=limit):
                user_data["posts"].append(post)
        except Exception as e:
            print(f"Post fetch error for {username}: {e}")

        try:
            for comment in user.comments.new(limit=limit):
                user_data["comments"].append(comment)
        except Exception as e:
            print(f"Comment fetch error for {username}: {e}")

        return user_data

    except Exception as e:
        print(f"Error fetching user {username}: {e}")
        return None

def compute_features_natural(user_data):
    """Compute raw heuristic features for bot likelihood."""
    now = time.time()
    age_days = (now - user_data["created_utc"]) / (60 * 60 * 24)
    posts = user_data["posts"]
    comments = user_data["comments"]

    # --- 1. Account age (in days) ---
    age = age_days

    # --- 2. Comment-to-post ratio ---
    total_posts = max(len(posts), 1)
    total_comments = len(comments)
    comment_to_post_ratio = total_comments / total_posts

    # --- 3. Subreddit diversity (number of unique subreddits) ---
    subreddits = (
        [p.subreddit.display_name for p in posts] +
        [c.subreddit.display_name for c in comments]
    )
    subreddit_diversity = len(set(subreddits))

    # --- 4. Activity spikes (boolean: 1 = spike detected, 0 = none) ---
    times = sorted([x.created_utc for x in posts + comments])
    activity_spike = 0
    if len(times) > 5:
        deltas = np.diff(times)
        mean_gap = np.mean(deltas)
        if np.any((deltas[:-1] > mean_gap * 5) & (deltas[1:] < mean_gap / 5)):
            activity_spike = 1

    # --- 5. Post-to-karma ratio ---
    # If available, try to fetch karma dynamically
    total_karma = 0
    try:
        user = reddit.redditor(user_data["username"])
        total_karma = user.link_karma + user.comment_karma
    except Exception:
        total_karma = 0

    post_to_karma_ratio = total_posts / max(total_karma, 1)


    # --- 7. Posts per day (raw count per day) ---
    posts_per_day = (len(posts) + len(comments)) / max(age_days, 1)

    return {
        "age_days": age,
        "total_posts": total_posts,
        "total_comments": total_comments,
        "comment_to_post_ratio": comment_to_post_ratio,
        "subreddit_diversity": subreddit_diversity,
        "activity_spike": activity_spike,
        "post_to_karma_ratio": post_to_karma_ratio,
        "posts_per_day": posts_per_day,
    }

def analyze_user(username):
    data = fetch_user_data_safe(username)
    if data is None:
        return None
    try:
        return compute_features_natural(data)
    except Exception as e:
        print(f"Error for {username}: {e}")
        return None
    

if __name__ == "__main__":
    parquet_file1 = "users.parquet"
    parquet_file2 = "user_scores.parquet"

    if not os.path.exists(parquet_file1):
        # Create an empty DataFrame with the columns you expect
        columns = columns = [
                    'username',
                    'age_days',
                    'total_posts',
                    'total_comments',
                    'comment_to_post_ratio',
                    'subreddit_diversity',
                    'activity_spike',
                    'post_to_karma_ratio',
                    'posts_per_day'
                ]
        empty_df = pd.DataFrame(columns=columns)
        
        # Save it as a Parquet file
        empty_df.to_parquet(parquet_file1, engine="pyarrow", index=False)
        print(f"Created empty Parquet file at {parquet_file1}")

    if not os.path.exists(parquet_file2):
        # Create an empty DataFrame with the columns you expect
        columns = ['username','score']  # add any other needed columns
        empty_df = pd.DataFrame(columns=columns)
        
        # Save it as a Parquet file
        empty_df.to_parquet(parquet_file2, engine="pyarrow", index=False)
        print(f"Created empty Parquet file at {parquet_file2}")


    test_user = input("Enter Reddit username to analyze: ")
    generate_user_table([test_user])
    classify_bots([test_user])



