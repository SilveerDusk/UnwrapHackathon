import requests

headers = {"User-Agent": "my_reddit_app:v1.0 (by u/YOUR_USERNAME)"}
#the limit seems to be capped at 100, so we can only get 100 posts at a time
url = "https://www.reddit.com/r/uberdrivers/new.json?limit=10000"
response = requests.get(url, headers=headers)
data = response.json()

for post in data["data"]["children"]:
    print(post["data"]["title"])

print("Total Number of Posts Fetched: ", len(data["data"]["children"]))

