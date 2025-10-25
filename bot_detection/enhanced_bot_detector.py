#!/usr/bin/env python3
"""
Enhanced Bot Detection System with 0-100 Scoring
Based on behavioral analysis and pattern recognition
"""

import praw
from dotenv import load_dotenv
import os
import numpy as np
import time
from datetime import datetime
from collections import Counter
import math
import json
import re
from difflib import SequenceMatcher

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="bot detection unwrapathon"
)

def fetch_user_data(username, limit=50):
    """Fetch user metadata, latest posts and comments."""
    user = reddit.redditor(username)

    user_data = {
        "username": username,
        "created_utc": user.created_utc,
        "link_karma": user.link_karma,
        "comment_karma": user.comment_karma,
        "is_verified": user.verified if hasattr(user, 'verified') else False,
        "posts": [],
        "comments": []
    }

    for post in user.submissions.new(limit=limit):
        user_data["posts"].append({
            "created_utc": post.created_utc,
            "subreddit": str(post.subreddit),
            "score": post.score,
            "title": post.title,
            "selftext": post.selftext
        })
    
    for com in user.comments.new(limit=limit):
        user_data["comments"].append({
            "created_utc": com.created_utc,
            "subreddit": str(com.subreddit),
            "score": com.score,
            "body": com.body
        })
    
    return user_data


def clean_text(text: str) -> str:
    """Basic text normalization for Reddit posts/comments."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_duplicate_content_ratio(items, content_key):
    """Compute ratio of near-duplicate comments/titles (0-1)."""
    texts = []
    for item in items:
        if isinstance(item, dict) and content_key in item:
            text = item[content_key]
            if text and text.strip():
                texts.append(clean_text(text))
    
    if len(texts) < 3:
        return 0.0
    
    duplicate_count = 0
    total_comparisons = 0
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
            total_comparisons += 1
            if sim > 0.8:
                duplicate_count += 1
    
    return duplicate_count / max(total_comparisons, 1)


def compute_features(user_data):
    """Compute comprehensive features for bot detection."""
    now = time.time()
    age_days = (now - user_data["created_utc"]) / (60 * 60 * 24)
    posts = user_data["posts"]
    comments = user_data["comments"]

    age_score = np.exp(-age_days / 180)

    c_to_p = len(comments) / (len(posts) + 1e-6)
    comment_to_post_score = 1 - min(1.0, c_to_p) if c_to_p < 1 else 0.0

    subreddits = [p["subreddit"] for p in posts] + [c["subreddit"] for c in comments]
    subreddit_count = len(set(subreddits))
    subreddit_diversity_score = 1 - min(subreddit_count / 10, 1.0)

    times = sorted([x["created_utc"] for x in posts + comments])
    activity_spike_score = 0.0
    if len(times) > 5:
        deltas = np.diff(times)
        mean_gap = np.mean(deltas)
        if mean_gap > 0:
            if np.any((deltas[:-1] > mean_gap * 5) & (deltas[1:] < mean_gap / 5)):
                activity_spike_score = 1.0

    total_posts = len(posts) + len(comments)
    total_karma = user_data["link_karma"] + user_data["comment_karma"]
    post_to_karma_ratio = total_posts / (total_karma + 1e-6)
    post_to_karma_score = min(post_to_karma_ratio * 10, 1.0)

    comment_dupe_ratio = get_duplicate_content_ratio(comments, "body")
    post_dupe_ratio = get_duplicate_content_ratio(posts, "title")
    duplicate_content_score = max(comment_dupe_ratio, post_dupe_ratio)

    posts_per_day = total_posts / max(age_days, 1)
    posts_per_day_score = min(posts_per_day / 20, 1.0)

    username = user_data["username"]
    username_suspicious = 0.0
    
    if re.match(r'^\w+_\w+\d{4}$', username):
        username_suspicious = 0.5
    elif re.match(r'^[a-z]{8,}\d{4,}$', username.lower()):
        username_suspicious = 0.7
    elif len(username) > 20 or (len(username) < 4 and not username.isalpha()):
        username_suspicious = 0.3

    avg_karma_per_post = total_karma / max(total_posts, 1)
    low_karma_score = 1.0 if avg_karma_per_post < 2 else 0.0

    verification_penalty = 0.0 if user_data.get("is_verified", False) else 0.2

    return {
        "age_score": age_score,
        "age_days": age_days,
        "comment_to_post_score": comment_to_post_score,
        "subreddit_diversity_score": subreddit_diversity_score,
        "subreddit_count": subreddit_count,
        "activity_spike_score": activity_spike_score,
        "post_to_karma_score": post_to_karma_score,
        "duplicate_content_score": duplicate_content_score,
        "posts_per_day_score": posts_per_day_score,
        "username_suspicious_score": username_suspicious,
        "low_karma_score": low_karma_score,
        "verification_penalty": verification_penalty,
        "total_posts": len(posts),
        "total_comments": len(comments),
        "total_karma": total_karma,
        "avg_karma_per_post": avg_karma_per_post
    }


def compute_bot_score_100_enhanced(features):
    """
    Enhanced bot scoring with explicit weights (0-100 scale)
    
    Scoring breakdown:
    - Account Age: 0-20 points
    - Activity Pattern: 0-20 points
    - Content Quality: 0-20 points
    - Engagement: 0-20 points
    - Diversity: 0-20 points
    """
    
    total_score = 0.0
    breakdown = {}
    
    age_penalty = 20 * min(1.0, math.exp(-features["age_days"] / 90))
    total_score += age_penalty
    breakdown["account_age_penalty"] = round(age_penalty, 2)
    
    activity_score = 0.0
    
    if features["activity_spike_score"] > 0:
        activity_score += 10
    
    if features["posts_per_day_score"] > 0.5:
        activity_score += 10 * features["posts_per_day_score"]
    
    activity_score += 5 * features["username_suspicious_score"]
    
    total_score += min(activity_score, 20)
    breakdown["activity_pattern_penalty"] = round(min(activity_score, 20), 2)
    
    content_score = 0.0
    
    content_score += 15 * features["duplicate_content_score"]
    content_score += 5 * features["low_karma_score"]
    
    total_score += min(content_score, 20)
    breakdown["content_quality_penalty"] = round(min(content_score, 20), 2)
    
    engagement_score = 20 * min(features["post_to_karma_score"] * 2, 1.0)
    total_score += engagement_score
    breakdown["engagement_penalty"] = round(engagement_score, 2)
    
    diversity_score = 0.0
    
    diversity_score += 15 * features["subreddit_diversity_score"]
    diversity_score += 5 * features["comment_to_post_score"]
    
    total_score += min(diversity_score, 20)
    breakdown["diversity_penalty"] = round(min(diversity_score, 20), 2)
    
    total_score += features["verification_penalty"] * 5
    breakdown["verification_penalty"] = round(features["verification_penalty"] * 5, 2)
    
    return round(min(total_score, 100), 2), breakdown


def classify_bot_likelihood(bot_score_100):
    """Classify users based on bot score (0-100)"""
    if bot_score_100 < 30:
        return {
            "classification": "Likely Human",
            "confidence": "High",
            "color": "green",
            "description": "Normal user behavior patterns detected",
            "risk_level": "Low"
        }
    elif bot_score_100 < 50:
        return {
            "classification": "Possibly Suspicious",
            "confidence": "Medium",
            "color": "yellow",
            "description": "Some bot-like characteristics detected",
            "risk_level": "Medium"
        }
    elif bot_score_100 < 70:
        return {
            "classification": "Likely Bot",
            "confidence": "Medium-High",
            "color": "orange",
            "description": "Multiple bot indicators present",
            "risk_level": "High"
        }
    else:
        return {
            "classification": "Almost Certainly Bot",
            "confidence": "Very High",
            "color": "red",
            "description": "Strong bot behavior patterns detected",
            "risk_level": "Critical"
        }


def analyze_user_comprehensive(username):
    """
    Comprehensive user analysis with detailed bot score breakdown.
    Returns complete analysis in JSON-serializable format.
    """
    try:
        data = fetch_user_data(username)
        features = compute_features(data)
        bot_score_100, breakdown = compute_bot_score_100_enhanced(features)
        classification = classify_bot_likelihood(bot_score_100)
        
        result = {
            "username": username,
            "analyzed_at": datetime.now().isoformat(),
            "bot_score": bot_score_100,
            "classification": classification["classification"],
            "confidence": classification["confidence"],
            "risk_level": classification["risk_level"],
            "description": classification["description"],
            "breakdown": breakdown,
            "account_info": {
                "account_age_days": round(features["age_days"], 1),
                "total_posts": features["total_posts"],
                "total_comments": features["total_comments"],
                "total_karma": features["total_karma"],
                "subreddit_count": features["subreddit_count"],
                "avg_karma_per_post": round(features["avg_karma_per_post"], 2),
                "posts_per_day": round(features["total_posts"] / max(features["age_days"], 1), 2)
            },
            "red_flags": generate_red_flags(features, bot_score_100),
            "recommendations": generate_recommendations(bot_score_100, features)
        }
        
        return result
        
    except Exception as e:
        return {
            "username": username,
            "error": str(e),
            "bot_score": None,
            "classification": "Error",
            "analyzed_at": datetime.now().isoformat()
        }


def generate_red_flags(features, bot_score):
    """Generate list of detected red flags."""
    red_flags = []
    
    if features["age_days"] < 90:
        red_flags.append(f"Very new account ({round(features['age_days'], 1)} days old)")
    
    if features["subreddit_count"] < 3:
        red_flags.append(f"Limited subreddit activity (only {features['subreddit_count']} subreddit(s))")
    
    if features["activity_spike_score"] > 0:
        red_flags.append("Unusual activity spikes detected")
    
    if features["duplicate_content_score"] > 0.3:
        red_flags.append(f"High duplicate content ({round(features['duplicate_content_score'] * 100, 1)}% similarity)")
    
    if features["posts_per_day_score"] > 0.7:
        posts_per_day = features["total_posts"] / max(features["age_days"], 1)
        red_flags.append(f"Extremely high posting frequency ({round(posts_per_day, 1)} posts/day)")
    
    if features["avg_karma_per_post"] < 2:
        red_flags.append(f"Very low engagement (avg {round(features['avg_karma_per_post'], 2)} karma per post)")
    
    if features["username_suspicious_score"] > 0.5:
        red_flags.append("Username follows auto-generated pattern")
    
    if not red_flags:
        red_flags.append("No major red flags detected")
    
    return red_flags


def generate_recommendations(bot_score, features):
    """Generate actionable recommendations based on score."""
    if bot_score < 30:
        return "Account appears legitimate. No action needed."
    elif bot_score < 50:
        return "Monitor account activity. Consider manual review if behavior persists."
    elif bot_score < 70:
        return "High probability of bot activity. Recommend flagging for review and possible restrictions."
    else:
        return "Very high probability of bot activity. Immediate action recommended: ban or severe restrictions."


def analyze_multiple_users(usernames, save_to_file=True):
    """Analyze multiple users and optionally save results to JSON."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Analyzing {len(usernames)} users...")
    print(f"{'='*60}\n")
    
    for i, username in enumerate(usernames, 1):
        print(f"[{i}/{len(usernames)}] Analyzing u/{username}...", end=" ")
        result = analyze_user_comprehensive(username)
        results.append(result)
        
        if result.get("bot_score") is not None:
            print(f"Score: {result['bot_score']}/100")
        else:
            print(f"Error: {result.get('error', 'Unknown')}")
    
    valid_scores = [r["bot_score"] for r in results if r.get("bot_score") is not None]
    
    if valid_scores:
        stats = {
            "total_analyzed": len(usernames),
            "successful_analyses": len(valid_scores),
            "failed_analyses": len(usernames) - len(valid_scores),
            "average_bot_score": round(np.mean(valid_scores), 2),
            "median_bot_score": round(np.median(valid_scores), 2),
            "min_bot_score": round(min(valid_scores), 2),
            "max_bot_score": round(max(valid_scores), 2),
            "likely_humans": sum(1 for s in valid_scores if s < 30),
            "suspicious": sum(1 for s in valid_scores if 30 <= s < 50),
            "likely_bots": sum(1 for s in valid_scores if 50 <= s < 70),
            "almost_certain_bots": sum(1 for s in valid_scores if s >= 70)
        }
        
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Analyzed: {stats['total_analyzed']}")
        print(f"Successful: {stats['successful_analyses']}")
        print(f"Failed: {stats['failed_analyses']}")
        print(f"\nBot Score Statistics:")
        print(f"  Average: {stats['average_bot_score']}/100")
        print(f"  Median: {stats['median_bot_score']}/100")
        print(f"  Range: {stats['min_bot_score']}-{stats['max_bot_score']}")
        print(f"\nClassification Breakdown:")
        print(f"  Likely Humans: {stats['likely_humans']} ({stats['likely_humans']/stats['successful_analyses']*100:.1f}%)")
        print(f"  Suspicious: {stats['suspicious']} ({stats['suspicious']/stats['successful_analyses']*100:.1f}%)")
        print(f"  Likely Bots: {stats['likely_bots']} ({stats['likely_bots']/stats['successful_analyses']*100:.1f}%)")
        print(f"  Almost Certain Bots: {stats['almost_certain_bots']} ({stats['almost_certain_bots']/stats['successful_analyses']*100:.1f}%)")
        print(f"{'='*60}\n")
        
        output = {
            "analysis_date": datetime.now().isoformat(),
            "statistics": stats,
            "results": results
        }
        
        if save_to_file:
            filename = f"bot_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to: {filename}\n")
        
        return output
    
    return {"results": results}


if __name__ == "__main__":
    print("Enhanced Reddit Bot Detector (0-100 Scale)")
    print("=" * 60)
    
    mode = input("\nChoose mode:\n  1. Single user analysis\n  2. Multiple users analysis\n\nEnter choice (1 or 2): ").strip()
    
    if mode == "1":
        username = input("\nEnter Reddit username to analyze: ").strip()
        print(f"\nAnalyzing u/{username}...\n")
        
        result = analyze_user_comprehensive(username)
        
        if result.get("bot_score") is not None:
            print("=" * 60)
            print(f"{result['username'].upper()} - BOT DETECTION REPORT")
            print("=" * 60)
            
            classification = classify_bot_likelihood(result["bot_score"])
            print(f"\nBOT SCORE: {result['bot_score']}/100")
            print(f"Classification: {result['classification']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"\n{result['description']}")
            
            print(f"\nSCORE BREAKDOWN (Total: {result['bot_score']}/100)")
            print("-" * 60)
            for key, value in result["breakdown"].items():
                label = key.replace("_", " ").title()
                print(f"  {label:.<45} {value:>5.2f}")
            
            print(f"\nACCOUNT INFORMATION")
            print("-" * 60)
            for key, value in result["account_info"].items():
                label = key.replace("_", " ").title()
                print(f"  {label:.<45} {value}")
            
            print(f"\nRED FLAGS DETECTED")
            print("-" * 60)
            for flag in result["red_flags"]:
                print(f"  {flag}")
            
            print(f"\nRECOMMENDATION")
            print("-" * 60)
            print(f"  {result['recommendations']}")
            
            print("\n" + "=" * 60)
            
            filename = f"bot_analysis_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nDetailed report saved to: {filename}")
        else:
            print(f"Error analyzing user: {result.get('error', 'Unknown error')}")
    
    elif mode == "2":
        print("\nEnter usernames (comma-separated):")
        usernames_input = input("> ").strip()
        usernames = [u.strip() for u in usernames_input.split(",") if u.strip()]
        
        if usernames:
            analyze_multiple_users(usernames, save_to_file=True)
        else:
            print("No valid usernames provided")
    
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
