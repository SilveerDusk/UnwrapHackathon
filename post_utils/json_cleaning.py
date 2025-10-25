import json
from collections import defaultdict

def flatten_items(items, feature_counts, empty_counts):
  """Recursively flatten posts/comments and count features."""
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

def compute_empty_ratios(data):
  """Compute empty ratios for features in the JSON data."""
  feature_counts = defaultdict(int)
  empty_counts = defaultdict(int)

  # Top-level Listings
  for listing in data:
    children = listing.get("data", {}).get("children", [])
    flatten_items(children, feature_counts, empty_counts)

  empty_ratios = {k: empty_counts[k] / feature_counts[k] for k in feature_counts}
  return empty_ratios

def clean_item(item, features_to_remove):
  """Recursively remove unwanted fields from a post/comment."""
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
  """Clean the JSON data by removing unwanted fields."""
  for listing in data:
    children = listing.get("data", {}).get("children", [])
    for child in children:
      clean_item(child, features_to_remove)
  return data

def suggest_features_to_remove(empty_ratios, threshold=0.8):
  """Suggest features to remove based on empty ratios."""
  features_to_remove = [k for k, r in empty_ratios.items() if r > threshold]
  features_to_remove.extend([
    "body_html", "subreddit_name_prefixed", "subreddit_type", "can_mod_post", "gilded", "saved", "collapsed", "author_patreon_flair", "can_gild", "author_premium", "score_hidden", "no_follow", "author_flair_type", "total_awards_received", "subreddit", "send_replies"
  ])
  return features_to_remove

# Example usage functions

def load_json(file_path):
  """Load JSON data from a file."""
  with open(file_path) as f:
    return json.load(f)

def save_json(data, file_path):
  """Save JSON data to a file."""
  with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

__all__ = [
  'compute_empty_ratios',
  'suggest_features_to_remove',
  'clean_json',
  'load_json',
  'save_json'
]