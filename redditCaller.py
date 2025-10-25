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
    posts.append({
      "id": submission.id,
      "title": submission.title,
      "created_utc": submission.created_utc,
      "score": submission.score,
      "num_comments": submission.num_comments,
      "selftext": submission.selftext
    })

  return posts, submission.name  # return 'after' for pagination

def main():
  #add an arg for the number of posts to grab, round it up to a multiple of 100
  if len(sys.argv) != 2 or len(sys.argv[1]) != 3:
    print("Usage: python part1.py <number_of_posts> | <subreddit>")
    return
  number_of_posts = sys.argv[1]
  if len(sys.argv) == 3:
    subreddit = [sys.argv[2]]
  else:
    subreddit = ["uberdrivers"]

  number_of_posts = int(number_of_posts)
  batches = (number_of_posts + 99) // 100

  for subreddit in subreddit:
    print(f"Fetching {number_of_posts} posts from r/{subreddit}...")
    subreddit = reddit.subreddit("uberdrivers")
    posts = []
    for i in range(batches):
      if i == 0:
        after = None
      posts_batch, after = fetch_posts(subreddit, limit=100, after=after)
      posts.extend(posts_batch)
      if not after:
        break

    for post in posts:
      print(f"{post['title']} (Score: {post['score']}, Comments: {post['num_comments']})")
    print(f"Total posts fetched: {len(posts)}")

if __name__ == "__main__":
  main()