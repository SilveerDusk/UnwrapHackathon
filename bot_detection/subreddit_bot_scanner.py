#!/usr/bin/env python3
"""
Subreddit Bot Scanner
Collects users from specific subreddits and analyzes them for bot behavior
"""

import praw
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from enhanced_bot_detector import analyze_user_comprehensive, classify_bot_likelihood
import numpy as np

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="bot detection unwrapathon"
)


def collect_users_from_subreddit(subreddit_name, post_limit=50, comment_limit=100):
    """
    Collect unique usernames from a subreddit by scanning posts and comments.
    
    Args:
        subreddit_name: Name of the subreddit (e.g., 'uberdrivers')
        post_limit: Number of posts to scan
        comment_limit: Number of comments per post to scan
    
    Returns:
        set: Unique usernames found in the subreddit
    """
    print(f"\nCollecting users from r/{subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)
    usernames = set()
    
    try:
        for post in subreddit.hot(limit=post_limit):
            if post.author and post.author.name != "AutoModerator":
                usernames.add(post.author.name)
            
            try:
                post.comments.replace_more(limit=0)
                for comment in post.comments.list()[:comment_limit]:
                    if hasattr(comment, 'author') and comment.author and comment.author.name != "AutoModerator":
                        usernames.add(comment.author.name)
            except Exception as e:
                continue
        
        print(f"Found {len(usernames)} unique users")
        return usernames
    
    except Exception as e:
        print(f"Error collecting users from r/{subreddit_name}: {str(e)}")
        return set()


def collect_users_from_multiple_subreddits(subreddit_names, post_limit=50):
    """
    Collect users from multiple subreddits.
    
    Args:
        subreddit_names: List of subreddit names
        post_limit: Number of posts to scan per subreddit
    
    Returns:
        dict: Mapping of subreddit -> set of usernames
    """
    all_users = {}
    
    for subreddit_name in subreddit_names:
        users = collect_users_from_subreddit(subreddit_name, post_limit=post_limit)
        all_users[subreddit_name] = users
    
    total_unique = len(set().union(*all_users.values()))
    print(f"\nTotal unique users across all subreddits: {total_unique}")
    
    return all_users


def analyze_subreddit_users(subreddit_name, user_limit=20, post_limit=50):
    """
    Collect and analyze users from a specific subreddit.
    
    Args:
        subreddit_name: Name of the subreddit
        user_limit: Maximum number of users to analyze
        post_limit: Number of posts to scan for users
    
    Returns:
        dict: Analysis results with statistics
    """
    print(f"\n{'='*60}")
    print(f"SUBREDDIT BOT ANALYSIS: r/{subreddit_name}")
    print(f"{'='*60}")
    
    usernames = collect_users_from_subreddit(subreddit_name, post_limit=post_limit)
    
    if not usernames:
        print("No users found!")
        return None
    
    usernames_list = list(usernames)[:user_limit]
    print(f"\nAnalyzing {len(usernames_list)} users...")
    
    results = []
    
    for i, username in enumerate(usernames_list, 1):
        print(f"[{i}/{len(usernames_list)}] Analyzing u/{username}...", end=" ")
        
        try:
            result = analyze_user_comprehensive(username)
            results.append(result)
            
            if result.get("bot_score") is not None:
                classification = classify_bot_likelihood(result["bot_score"])
                print(f"Score: {result['bot_score']}/100 ({classification['classification']})")
            else:
                print(f"Error: {result.get('error', 'Unknown')}")
        
        except Exception as e:
            print(f"Error: {str(e)}")
            results.append({
                "username": username,
                "error": str(e),
                "bot_score": None
            })
    
    valid_scores = [r["bot_score"] for r in results if r.get("bot_score") is not None]
    
    if valid_scores:
        stats = {
            "subreddit": subreddit_name,
            "total_users_found": len(usernames),
            "users_analyzed": len(usernames_list),
            "successful_analyses": len(valid_scores),
            "failed_analyses": len(usernames_list) - len(valid_scores),
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
        print(f"ANALYSIS SUMMARY - r/{subreddit_name}")
        print(f"{'='*60}")
        print(f"Total Users Found: {stats['total_users_found']}")
        print(f"Users Analyzed: {stats['users_analyzed']}")
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
        
        filename = f"subreddit_analysis_{subreddit_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {filename}\n")
        
        return output
    
    return {"results": results}


def analyze_multiple_subreddits(subreddit_names, users_per_subreddit=20, posts_per_subreddit=50):
    """
    Analyze users from multiple subreddits and generate comparative report.
    
    Args:
        subreddit_names: List of subreddit names
        users_per_subreddit: Number of users to analyze per subreddit
        posts_per_subreddit: Number of posts to scan per subreddit
    
    Returns:
        dict: Combined analysis results
    """
    all_results = {}
    
    for subreddit_name in subreddit_names:
        result = analyze_subreddit_users(
            subreddit_name, 
            user_limit=users_per_subreddit, 
            post_limit=posts_per_subreddit
        )
        all_results[subreddit_name] = result
    
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS ACROSS SUBREDDITS")
    print(f"{'='*60}\n")
    
    for subreddit_name, data in all_results.items():
        if data and data.get('statistics'):
            stats = data['statistics']
            print(f"r/{subreddit_name}:")
            print(f"  Average Bot Score: {stats['average_bot_score']}/100")
            print(f"  Bot Prevalence: {(stats['likely_bots'] + stats['almost_certain_bots'])/stats['successful_analyses']*100:.1f}%")
            print(f"  Suspicious+: {(stats['suspicious'] + stats['likely_bots'] + stats['almost_certain_bots'])/stats['successful_analyses']*100:.1f}%")
            print()
    
    filename = f"multi_subreddit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to: {filename}\n")
    
    return all_results


if __name__ == "__main__":
    print("Subreddit Bot Scanner")
    print("=" * 60)
    
    mode = input("\nChoose mode:\n  1. Analyze single subreddit\n  2. Analyze multiple subreddits\n  3. Quick test (analyze 5 users from a subreddit)\n\nEnter choice (1, 2, or 3): ").strip()
    
    if mode == "1":
        subreddit = input("\nEnter subreddit name (without r/): ").strip()
        user_limit = input("Number of users to analyze (default 20): ").strip()
        user_limit = int(user_limit) if user_limit else 20
        
        analyze_subreddit_users(subreddit, user_limit=user_limit)
    
    elif mode == "2":
        subreddits_input = input("\nEnter subreddit names (comma-separated, without r/): ").strip()
        subreddits = [s.strip() for s in subreddits_input.split(",") if s.strip()]
        
        user_limit = input("Number of users per subreddit (default 20): ").strip()
        user_limit = int(user_limit) if user_limit else 20
        
        if subreddits:
            analyze_multiple_subreddits(subreddits, users_per_subreddit=user_limit)
        else:
            print("No valid subreddits provided")
    
    elif mode == "3":
        subreddit = input("\nEnter subreddit name (without r/): ").strip()
        analyze_subreddit_users(subreddit, user_limit=5, post_limit=20)
    
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")

