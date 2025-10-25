import requests
import sys
import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
  client_id=os.getenv("CLIENT_ID"),
  client_secret=os.getenv("CLIENT_SECRET"),
  user_agent="my_reddit_app:v1.0 (by u/SilveerDusk)"
)

def fetch_posts(subreddit, limit=100, after=None):
  """Fetch latest posts from a subreddit."""
  posts = []

  for submission in subreddit.new(limit=limit, params={"after": None}):
    post_id = submission.id
    post_title = submission.title
    post_creator = submission.created_utc
    post_score = submission.score
    post_num_comments = submission.num_comments
    selftext = submission.selftext
    subreddit_name = submission.subreddit.display_name
    author_name = submission.author.name if submission.author else "[deleted]"
    url = submission.url
    stickied = submission.stickied

    posts.append({
      "id": post_id,
      "title": post_title,
      "selftext": selftext,
      "subreddit": subreddit_name,
      "created_utc": post_creator,
      "score": post_score,
      "num_comments": post_num_comments,
      "author": author_name,
      "url": url,
      "stickied": stickied
    })

  return posts, submission.name  # return 'after' for pagination

def fetch_comments(post_id):
  """Fetch comments for a given post ID."""
  submission = reddit.submission(id=post_id)
  submission.comments.replace_more(limit=0)
  comments = []

  for comment in submission.comments.list():
    comment_id = comment.id
    comment_body = comment.body
    post_id = comment.link_id
    created_utc = comment.created_utc
    score = comment.score
    comment_author = comment.author.name if comment.author else "[deleted]"
    url = comment.permalink
    is_submitter = comment.is_submitter

    comments.append({
      "id": comment_id,
      "body": comment_body,
      "post_id": post_id,
      "created_utc": created_utc,
      "score": score,
      "author": comment_author,
      "url": url,
      "is_submitter": is_submitter
    })

  return comments


def fetch_subreddit_posts(subreddit_name="uberdrivers", number_of_posts=100):
  """
  Fetch posts from a subreddit with pagination support.
  
  Args:
    subreddit_name (str): Name of the subreddit to fetch posts from
    number_of_posts (int): Number of posts to fetch
    
  Returns:
    list: List of post dictionaries
  """
  batches = (number_of_posts + 99) // 100
  subreddit = reddit.subreddit(subreddit_name)
  posts = []
  after = None
  
  print(f"Fetching {number_of_posts} posts from r/{subreddit_name}...")
  
  for i in range(batches):
    posts_batch, after = fetch_posts(subreddit, limit=100, after=after)
    posts.extend(posts_batch)
    if not after:
      break
      
  print(f"Total posts fetched: {len(posts)}")

  return posts

def main():
  if len(sys.argv) < 2:
    print("Usage: python redditCaller.py <number_of_posts> [subreddit]")
    return
    
  number_of_posts = int(sys.argv[1])
  subreddit_name = sys.argv[2] if len(sys.argv) > 2 else "uberdrivers"

  posts = fetch_subreddit_posts(subreddit_name, number_of_posts)

  for post in posts:
    print(f"{post['title']} (Score: {post['score']}, Comments: {post['num_comments']})")

if __name__ == "__main__":
  main()

  __all__ = ['fetch_posts', 'fetch_comments', 'fetch_subreddit_posts']