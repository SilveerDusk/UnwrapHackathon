import requests

headers = {"User-Agent": "my_reddit_app:v1.0 (by u/YOUR_USERNAME)"}
url = "https://www.reddit.com/r/uberdrivers/top.json?limit=5"
response = requests.get(url, headers=headers)
data = response.json()

for post in data["data"]["children"]:
    print(post["data"]["title"])

