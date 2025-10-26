import os
import sys
import asyncio
import ast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils.database import RedditDataManager
from sklearn.metrics.pairwise import cosine_similarity
from unwrap_openai.unwrap_openai import summarize_post, summarize_comments
from tqdm import tqdm

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

async def get_insights_for_subreddit(subreddit):
  subreddit_posts = dbManager.get_all_posts(subreddit)
  print(f"Fetched {len(subreddit_posts)} posts from r/uberdrivers for insight creation")

  post_summaries, post_comments_summaries, mentions = await create_insights(subreddit_posts)
  print(post_summaries, post_comments_summaries)
  insights = {}
  for i in range(len(post_summaries)):
    #print(post_summaries[i])
    for summary in post_summaries[i]:
      if summary not in insights:
        insights[summary] = {
          "mentions": [],
        }
      insights[summary]["mentions"].append(mentions[i])
  print(f"Generated {len(insights)} unique insights from posts and comments")
  formatted_insights = []
  for insight, data in insights.items():
    formatted_insights.append({
      "insight": insight,
      "mentions": data["mentions"],
      "num_mentions": len(data["mentions"]),
    })
  print(formatted_insights)
  return formatted_insights

async def main():
  """Main function - complete pipeline for fetching, processing, and storing Reddit data"""
  for subreddit in ["uberdrivers"]:
    raw_insights = await get_insights_for_subreddit(subreddit)
    filtered_insights = raw_insights
    data_for_db = {
      "subreddit": subreddit,
      "raw_insights": raw_insights,
      "filtered_insights": filtered_insights,
    }
    dbManager.insert_insight(data_for_db)

if __name__ == "__main__":
  asyncio.run(main())