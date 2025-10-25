import requests
import sys
import praw
import os
import json
import argparse
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Add the parent directory to the path to import database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils.database import RedditDataManager

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

reddit = praw.Reddit(
  client_id=os.getenv("CLIENT_ID"),
  client_secret=os.getenv("CLIENT_SECRET"),
  user_agent="my_reddit_app:v1.0 (by u/SilveerDusk)"
)

class RedditCaller:
    """
    Reddit data fetcher and processor with embedding generation and MongoDB insertion.
    """
    
    def __init__(self):
        """Initialize database connection and embedding model"""
        self.db_manager = RedditDataManager()
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        self.db_manager.mongo.close()
        logger.info("Database connection closed")
    
    def fetch_posts(self, subreddit, limit=100, after=None):
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

    def fetch_comments(self, post_id):
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

    def fetch_subreddit_posts(self, subreddit_name="uberdrivers", number_of_posts=100):
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
        
        for _ in range(batches):
            posts_batch, after = self.fetch_posts(subreddit, limit=100, after=after)
            posts.extend(posts_batch)
            if not after:
                break
                
        print(f"Total posts fetched: {len(posts)}")

        return posts
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence transformers"""
        try:
            if not text or text.strip() == "":
                return [0.0] * 384  # Return zero vector for empty text
            
            # Truncate very long texts to avoid memory issues
            if len(text) > 1000:
                text = text[:1000]
            
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 384
    
    def process_post(self, post_data: Dict) -> Dict:
        """Process a single post: add metadata and generate embedding"""
        try:
            # Generate embedding for post (title + selftext)
            post_text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}".strip()
            embedding = self.generate_embedding(post_text)
            
            # Add required fields and metadata
            processed_post = {
                "id": post_data.get("id"),
                "title": post_data.get("title", ""),
                "selftext": post_data.get("selftext", ""),
                "subreddit": post_data.get("subreddit", ""),
                "created_utc": post_data.get("created_utc"),
                "score": post_data.get("score", 0),
                "num_comments": post_data.get("num_comments", 0),
                "author": post_data.get("author", ""),
                "url": post_data.get("url", ""),
                "stickied": post_data.get("stickied", False),
                "embedding": embedding
            }
            
            return processed_post
            
        except Exception as e:
            logger.error(f"Failed to process post {post_data.get('id', 'unknown')}: {e}")
            raise
    
    def process_comment(self, comment_data: Dict) -> Dict:
        """Process a single comment: add metadata and generate embedding"""
        try:
            # Generate embedding for comment body
            comment_text = comment_data.get("body", "")
            embedding = self.generate_embedding(comment_text)
            
            # Add required fields and metadata
            processed_comment = {
                "id": comment_data.get("id"),
                "body": comment_data.get("body", ""),
                "post_id": comment_data.get("post_id"),
                "created_utc": comment_data.get("created_utc"),
                "score": comment_data.get("score", 0),
                "author": comment_data.get("author", ""),
                "is_submitter": comment_data.get("is_submitter", False),
                "embedding": embedding
            }
            
            return processed_comment
            
        except Exception as e:
            logger.error(f"Failed to process comment {comment_data.get('id', 'unknown')}: {e}")
            raise

__all__ = ['RedditCaller']