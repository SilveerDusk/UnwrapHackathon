import requests

headers = {"User-Agent": "my_reddit_app:v1.0 (by u/YOUR_USERNAME)"}
#the limit seems to be capped at 100, so we can only get 100 posts at a time
url = "https://www.reddit.com/r/uberdrivers/new.json?limit=10000&after=t3_1oevibm"
response = requests.get(url, headers=headers)
data = response.json()

for post in data["data"]["children"]:
    print(post["data"]["title"])

print("Total Number of Posts Fetched: ", len(data["data"]["children"]))
print(data["data"]["after"])

def fetch_posts(subreddit, limit=100):
  """Fetch latest posts from a subreddit."""
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
    
  return posts

