import os
import sys
import asyncio
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils.database import RedditDataManager
from unwrap_openai.unwrap_openai import summarize_post, summarize_comments, generalize_insights
from tqdm import tqdm
import pandas as pd

dbManager = RedditDataManager()

async def create_insights(post_data):
  post_summaries = []
  post_comments_summaries = []
  mentions = []
  for post in tqdm(post_data):
    post_id = post.get("id")
    title = post.get("title")
    selftext = post.get("selftext")
    subreddit = post.get("subreddit")
    created_utc = post.get("created_utc")
    score = post.get("score")
    num_comments = post.get("num_comments")
    author = post.get("author")
    url = post.get("url")
    stickied = post.get("stickied")
    embedding = post.get("embedding")
    created_at = post.get("created_at")
    inserted_at = post.get("inserted_at")
    mentions.append({
      "post_id": post_id,
      "post_title": title,
      "post_body": selftext,
      "data_posted": created_utc,
      "score": score,
      "num_comments": num_comments,
      "author": author,
      "url": url,
    })
    post_comments = dbManager.get_all_comments_for_post(post_id)

    comment_embeddings = [comment.get("embedding") for comment in post_comments if comment.get("embedding")]
    similarity_score = cosine_similarity(comment_embeddings, [embedding]) if comment_embeddings else []

    average_similarity = similarity_score.mean() if len(similarity_score) > 0 else 0.0
    #print(f"Average similarity between comments and post: {average_similarity:.4f}")

    post_text = f"{title}\n\n{selftext}"
    post_summary = await summarize_post(post_text)
    try:
        parsed_summary = ast.literal_eval(post_summary)
        if isinstance(parsed_summary, list):
            post_summaries.append(parsed_summary)
        else:
            post_summaries.append([])
    except (ValueError, SyntaxError):
        post_summaries.append([])
    post_text = f"Post: {post_text}\n"
    if len(post_comments) > 0:
      comments_text = "\n".join([comment.get("body", "") for comment in post_comments])
      text = post_text + f"Comments:\n{comments_text}"
      comments_summary = await summarize_comments(text)
      post_comments_summaries.append(comments_summary)
    else:
      comments_summary = "N/A"
      post_comments_summaries.append(comments_summary)

    #print(f"Post Summary: {post_summary}")
    #print(f"Comments Summary: {comments_summary}")

  return post_summaries, post_comments_summaries, mentions

def group_similar_insights(insights, target_mentions=10):
    """Group similar insights using embeddings (or OpenAI) to achieve target mention count"""
    if len(insights) < 2:
        return [insights]
    
    # Extract insight texts and create embeddings
    embeddings = []
    valid_insights = []
    
    for i, insight in enumerate(insights):
        # Generate embedding for the insight text
        embedding = dbManager.generate_embedding(insight["insight"])
        if embedding and len(embedding) == 384:
            embeddings.append(embedding)
            valid_insights.append(insight)
    
    if len(embeddings) < 2:
        return [insights]
    
    embeddings_array = np.array(embeddings)
    
    # Calculate target number of groups (aim for 8-12 mentions per group)
    total_mentions = sum(insight["num_mentions"] for insight in valid_insights)
    target_groups = max(2, total_mentions // target_mentions)
    target_groups = min(target_groups, len(valid_insights) // 2)  # Don't over-cluster
    
    print(f"Grouping {len(valid_insights)} insights into ~{target_groups} groups (target: {target_mentions} mentions per group)")
    
    # Test different K values
    silhouette_scores = {}
    for k in range(2, min(target_groups + 3, len(valid_insights) // 2)):
        if k >= len(valid_insights):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        silhouette_avg = silhouette_score(embeddings_array, cluster_labels)
        silhouette_scores[k] = silhouette_avg
    
    # Choose best K
    best_k = max(silhouette_scores.items(), key=lambda x: x[1])[0] if silhouette_scores else 2
    best_k = max(2, min(best_k, target_groups))
    
    print(f"Using {best_k} groups for insight generalization")
    
    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Group insights by cluster
    groups = {}
    for i, insight in enumerate(valid_insights):
        cluster_id = cluster_labels[i]
        if cluster_id not in groups:
            groups[cluster_id] = []
        groups[cluster_id].append(insight)
    
    return list(groups.values())

async def generalize_insight_group(insight_group):
    """Use OpenAI to generalize a group of similar insights into one broader insight"""
    if not insight_group:
        return None
    
    # Prepare insight data for OpenAI
    insight_data = []
    all_mentions = []
    
    for insight in insight_group:
        insight_data.append({
            "insight": insight["insight"],
            "mentions": insight["num_mentions"]
        })
        all_mentions.extend(insight["mentions"])
    
    # Use OpenAI to generalize the insights
    try:
        generalized_insight = await generalize_insights(insight_data)
        
        return {
            "insight": generalized_insight,
            "mentions": all_mentions,
            "num_mentions": len(all_mentions),
            "original_insights": [insight["insight"] for insight in insight_group]
        }
    except Exception as e:
        print(f"Error generalizing insights: {e}")
        # Fallback: combine insights manually
        combined_insight = " / ".join([insight["insight"] for insight in insight_group[:3]])
        return {
            "insight": combined_insight,
            "mentions": all_mentions,
            "num_mentions": len(all_mentions),
            "original_insights": [insight["insight"] for insight in insight_group]
        }

async def get_insights_for_subreddit(subreddit):
  """Create broader insights by generalizing post-specific insights"""
  subreddit_posts = dbManager.get_all_posts(subreddit)
  print(f"Fetched {len(subreddit_posts)} posts from r/{subreddit} for insight creation")
  
  if len(subreddit_posts) < 10:
    print("Not enough posts for analysis. Need at least 10 posts.")
    return []
  
  # Step 1: Generate specific insights for each post
  print("Step 1: Generating post-specific insights...")
  post_insights = await create_post_specific_insights(subreddit_posts)
  print(f"Generated {len(post_insights)} post-specific insights")
  
  # Step 2: Group similar insights using embeddings
  print("Step 2: Grouping similar insights...")
  insight_groups = group_similar_insights(post_insights)
  print(f"Grouped into {len(insight_groups)} insight groups")
  
  # Step 3: Generalize each group using ChatGPT
  print("Step 3: Generalizing insights with ChatGPT...")
  generalized_insights = await generalize_insight_groups(insight_groups)
  
  # Step 4: Filter and validate insights
  print("Step 4: Validating and filtering insights...")
  valid_insights = []
  for insight in generalized_insights:
    if insight["num_mentions"] >= 3:  # Minimum threshold
      valid_insights.append(insight)
  
  print(f"Generated {len(generalized_insights)} total insights")
  print(f"Valid insights (3+ mentions): {len(valid_insights)}")
  
  # Show statistics
  if valid_insights:
    mention_counts = [insight["num_mentions"] for insight in valid_insights]
    avg_mentions = sum(mention_counts) / len(mention_counts)
    print(f"Average mentions per insight: {avg_mentions:.1f}")
    print(f"Mention range: {min(mention_counts)} - {max(mention_counts)}")
    
    for insight in valid_insights:
      print(f"  - {insight['insight']}: {insight['num_mentions']} mentions")
  
  return valid_insights

async def create_post_specific_insights(posts):
  """Create specific insights for each post and group similar ones immediately"""
  all_insights = {}  # Dictionary to group insights by text
  
  for post in tqdm(posts):
    post_id = post.get("id")
    title = post.get("title")
    selftext = post.get("selftext")
    subreddit = post.get("subreddit")
    created_utc = post.get("created_utc")
    score = post.get("score")
    num_comments = post.get("num_comments")
    author = post.get("author")
    url = post.get("url")
    
    mention = {
      "post_id": post_id,
      "post_title": title,
      "post_body": selftext,
      "data_posted": created_utc,
      "score": score,
      "num_comments": num_comments,
      "author": author,
      "url": url,
    }
    
    post_comments = dbManager.get_all_comments_for_post(post_id)
    
    post_text = f"{title}\n\n{selftext}"
    post_summary = await summarize_post(post_text)
    
    try:
      parsed_summary = ast.literal_eval(post_summary)
      if isinstance(parsed_summary, list):
        for summary in parsed_summary:
          # Group insights by exact text match
          if summary not in all_insights:
            all_insights[summary] = {
              "insight": summary,
              "mentions": [],
              "num_mentions": 0
            }
          all_insights[summary]["mentions"].append(mention)
          all_insights[summary]["num_mentions"] += 1
    except (ValueError, SyntaxError):
      pass  # Skip invalid summaries
  
  # Convert to list
  post_insights = list(all_insights.values())
  
  return post_insights

async def generalize_insight_groups(insight_groups):
  """Generalize all insight groups using OpenAI"""
  generalized_insights = []
  
  for i, group in enumerate(insight_groups):
    print(f"Generalizing group {i+1}/{len(insight_groups)} with {len(group)} insights...")
    generalized = await generalize_insight_group(group)
    if generalized:
      generalized_insights.append(generalized)
  
  return generalized_insights

def filter_raw_insights(raw_insights):
  user_scores = pd.read_parquet('../user_scores.parquet', engine="fastparquet")
  bot_users = user_scores[user_scores['score'] > 0.5]['username'].tolist()
  print(f"Bot Users: {bot_users}")

  filtered_insights = []
  
  for insight in raw_insights:
    # Create a deep copy of the insight
    filtered_insight = {
      "insight": insight["insight"],
      "mentions": [],
      "num_mentions": 0,
      "original_insights": insight.get("original_insights", [])
    }
    
    # Filter mentions
    for mention in insight['mentions']:
      if mention['author'] not in bot_users:
        filtered_insight['mentions'].append(mention)
    
    # Update mention count
    filtered_insight['num_mentions'] = len(filtered_insight['mentions'])
    
    # Only include insights that still have mentions after filtering
    if filtered_insight['num_mentions'] > 0:
      filtered_insights.append(filtered_insight)
  
  return filtered_insights


async def main():
  """Main function - complete pipeline for fetching, processing, and storing Reddit data"""
  for subreddit in ["uberdrivers"]:
    raw_insights = await get_insights_for_subreddit(subreddit)
    filtered_insights = filter_raw_insights(raw_insights)
    data_for_db = {
      "subreddit": subreddit,
      "raw_insights": raw_insights,
      "filtered_insights": filtered_insights,
    }
    dbManager.insert_insight(data_for_db)

if __name__ == "__main__":
  asyncio.run(main())