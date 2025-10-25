import requests

headers = {"User-Agent": "my_reddit_app:v1.0 (by u/SilveerDusk)"}

def fetch_posts(subreddit, limit=100, after=None):
  """Fetch latest posts from a subreddit."""
  if after:
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}&after={after}"
  else:
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
  response = requests.get(url, headers=headers)
  data = response.json()

  posts = []
  for post in data["data"]["children"]:
    posts.append({
      "id": post["data"]["id"],
      "title": post["data"]["title"],
      "created_utc": post["data"]["created_utc"],
      "score": post["data"]["score"],
      "num_comments": post["data"]["num_comments"]
    })

  return posts, data["data"]["after"]

def main():
  subreddit = "uberdrivers"

  posts = []
  for i in range(10):
    posts_batch, after = fetch_posts(subreddit, limit=100, after=None if i == 0 else after)
    posts.extend(posts_batch)
    if not after:
      break

  for post in posts:
    print(f"{post['title']} (Score: {post['score']}, Comments: {post['num_comments']})")

if __name__ == "__main__":
  main()