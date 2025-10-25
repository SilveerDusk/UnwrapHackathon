import json
from collections import defaultdict

# Load your JSON
with open("comment_example.json") as f:
    data = json.load(f)

# Flatten posts/comments recursively
def flatten_items(items, feature_counts, empty_counts):
  for item in items:
    d = item.get("data", {})
    for k, v in d.items():
      feature_counts[k] += 1
      if v in (None, "", [], {}):
        empty_counts[k] += 1
      
    # Recursively process comments
    if "replies" in d and isinstance(d["replies"], dict):
      replies = d["replies"].get("data", {}).get("children", [])
      flatten_items(replies, feature_counts, empty_counts)

feature_counts = defaultdict(int)
empty_counts = defaultdict(int)

# Top-level Listings
for listing in data:
  children = listing.get("data", {}).get("children", [])
  flatten_items(children, feature_counts, empty_counts)

# Compute empty ratios
empty_ratios = {k: empty_counts[k]/feature_counts[k] for k in feature_counts}

# List features and empty ratios
for k, r in sorted(empty_ratios.items(), key=lambda x: -x[1]):
  print(f"{k}: {r:.2f}")

# Suggested removal: >80% empty or irrelevant fields
features_to_remove = [k for k, r in empty_ratios.items() if r > 0.8]
print("\nFeatures to remove:", features_to_remove)


def clean_item(item, features_to_remove):
  """Recursively remove unwanted fields from a post/comment"""
  if "data" not in item:
    return

  # Remove keys
  for k in features_to_remove:
    if k in item["data"]:
      del item["data"][k]

  # Recursively clean replies (comments)
  replies = item["data"].get("replies")
  if isinstance(replies, dict) and "data" in replies:
    children = replies["data"].get("children", [])
    for child in children:
      clean_item(child, features_to_remove)

def clean_json(data, features_to_remove):
  for listing in data:
    children = listing.get("data", {}).get("children", [])
    for child in children:
      clean_item(child, features_to_remove)
  return data

# Example usage
cleaned_data = clean_json(data, features_to_remove)

with open("reddit_cleaned.json", "w") as f:
  json.dump(cleaned_data, f, indent=2)
