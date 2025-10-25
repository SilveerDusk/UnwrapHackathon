"""
Reddit Data Processor
====================

Unified file for processing Reddit data: scraping, embedding generation, and MongoDB insertion.
Uses the database connection from database.py for consistency.

Usage:
    # Process scraped data from file
    python reddit_processor.py --input scraped_data.json
    
    # Scrape and process directly
    python reddit_processor.py --scrape --subreddit AskReddit --max-posts 10
"""

import json
import argparse
import logging
import requests
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from database import RedditDataManager

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditProcessor:
    """
    Unified Reddit data processor: scraping, embedding, and insertion.
    """
    
    def __init__(self):
        """Initialize database connection and embedding model"""
        self.db_manager = RedditDataManager()
        self.embedding_model = None
        self.headers = {
            "User-Agent": "MisinformationDetector/1.0 (by u/your_username)"
        }
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
    
    def fetch_posts(self, subreddit: str, limit: int = 25, sort: str = "hot", days_back: int = 7) -> List[Dict]:
        """Fetch posts from a subreddit, optionally filtered by date"""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit * 2}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            posts = []
            cutoff_time = time.time() - (days_back * 24 * 60 * 60)  # 7 days ago
            
            for post_data in data["data"]["children"]:
                post = post_data["data"]
                
                # Filter by date (only posts from the last 7 days)
                if post["created_utc"] >= cutoff_time:
                    posts.append({
                        "id": post["id"],
                        "title": post["title"],
                        "selftext": post.get("selftext", ""),
                        "subreddit": subreddit,
                        "created_utc": post["created_utc"],
                        "score": post["score"],
                        "num_comments": post["num_comments"],
                        "author": post["author"],
                        "url": post.get("url", ""),
                        "permalink": f"https://reddit.com{post['permalink']}",
                        "is_self": post.get("is_self", False),
                        "over_18": post.get("over_18", False),
                        "stickied": post.get("stickied", False)
                    })
                    
                    # Stop when we have enough posts
                    if len(posts) >= limit:
                        break
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit} (last {days_back} days)")
            return posts
            
        except Exception as e:
            logger.error(f"Failed to fetch posts from r/{subreddit}: {e}")
            return []
    
    def fetch_comments_for_post(self, subreddit: str, post_id: str, limit: int = 50) -> List[Dict]:
        """Fetch comments for a specific post"""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit={limit}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            comments = []
            
            # The first item is the post, the second is the comments
            if len(data) > 1:
                comments_data = data[1]["data"]["children"]
                
                for comment_data in comments_data:
                    if comment_data["kind"] == "t1":  # t1 is a comment
                        comment = comment_data["data"]
                        # Skip deleted/removed comments
                        if comment.get("body") not in ["[deleted]", "[removed]"]:
                            comments.append({
                                "id": comment["id"],
                                "body": comment["body"],
                                "post_id": post_id,
                                "subreddit": subreddit,
                                "created_utc": comment["created_utc"],
                                "score": comment["score"],
                                "author": comment["author"],
                                "permalink": f"https://reddit.com{comment['permalink']}",
                                "is_submitter": comment.get("is_submitter", False)
                            })
            
            logger.info(f"Fetched {len(comments)} comments for post {post_id}")
            return comments
            
        except Exception as e:
            logger.error(f"Failed to fetch comments for post {post_id}: {e}")
            return []
    
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
                "permalink": post_data.get("permalink", ""),
                "is_self": post_data.get("is_self", False),
                "over_18": post_data.get("over_18", False),
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
                "subreddit": comment_data.get("subreddit", ""),
                "created_utc": comment_data.get("created_utc"),
                "score": comment_data.get("score", 0),
                "author": comment_data.get("author", ""),
                "permalink": comment_data.get("permalink", ""),
                "is_submitter": comment_data.get("is_submitter", False),
                "embedding": embedding
            }
            
            return processed_comment
            
        except Exception as e:
            logger.error(f"Failed to process comment {comment_data.get('id', 'unknown')}: {e}")
            raise
    
    def scrape_and_process(self, subreddit: str, max_posts: int = 10, max_comments_per_post: int = 20) -> Dict:
        """Scrape Reddit data and process it into MongoDB"""
        try:
            logger.info(f"Starting scrape and process of r/{subreddit}")
            
            # Fetch posts
            posts = self.fetch_posts(subreddit, limit=max_posts, sort="hot", days_back=7)
            if not posts:
                logger.warning(f"No posts found for r/{subreddit}")
                return {"posts": 0, "comments": 0, "errors": []}
            
            # Process and store posts
            stored_posts = 0
            stored_comments = 0
            errors = []
            
            for post in posts:
                try:
                    # Process post
                    processed_post = self.process_post(post)
                    
                    # Store post in database
                    self.db_manager.insert_post(processed_post)
                    stored_posts += 1
                    
                    # Fetch and process comments for this post
                    comments = self.fetch_comments_for_post(subreddit, post['id'], limit=max_comments_per_post)
                    
                    for comment in comments:
                        try:
                            # Process comment
                            processed_comment = self.process_comment(comment)
                            
                            # Store comment in database
                            self.db_manager.insert_comment(processed_comment)
                            stored_comments += 1
                            
                        except Exception as e:
                            error_msg = f"Failed to process comment {comment.get('id', 'unknown')}: {e}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                    
                    # Rate limiting - be respectful to Reddit's API
                    time.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Failed to process post {post.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            result = {
                "subreddit": subreddit,
                "posts": stored_posts,
                "comments": stored_comments,
                "errors": errors,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Scrape and process complete for r/{subreddit}: {stored_posts} posts, {stored_comments} comments")
            return result
            
        except Exception as e:
            logger.error(f"Failed to scrape and process r/{subreddit}: {e}")
            return {"posts": 0, "comments": 0, "errors": [str(e)]}
    
    def load_and_process(self, file_path: str, batch_size: int = 50) -> Dict:
        """Load scraped data from file and process it into MongoDB"""
        try:
            logger.info(f"Loading and processing scraped data from {file_path}")
            
            # Load scraped data
            with open(file_path, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            
            # Extract posts and comments
            posts = scraped_data.get("posts", [])
            comments = scraped_data.get("comments", [])
            
            logger.info(f"Found {len(posts)} posts and {len(comments)} comments to process")
            
            # Process and insert posts in batches
            posts_inserted = 0
            for i in range(0, len(posts), batch_size):
                batch_posts = posts[i:i + batch_size]
                processed_posts = [self.process_post(post) for post in batch_posts]
                self.db_manager.insert_posts_batch(processed_posts)
                posts_inserted += len(processed_posts)
                logger.info(f"Processed posts batch {i//batch_size + 1}/{(len(posts)-1)//batch_size + 1}")
            
            # Process and insert comments in batches
            comments_inserted = 0
            for i in range(0, len(comments), batch_size):
                batch_comments = comments[i:i + batch_size]
                processed_comments = [self.process_comment(comment) for comment in batch_comments]
                self.db_manager.insert_comments_batch(processed_comments)
                comments_inserted += len(processed_comments)
                logger.info(f"Processed comments batch {i//batch_size + 1}/{(len(comments)-1)//batch_size + 1}")
            
            result = {
                "posts": posts_inserted,
                "comments": comments_inserted,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Load and process complete: {posts_inserted} posts, {comments_inserted} comments")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load and process data: {e}")
            return {"posts": 0, "comments": 0, "errors": [str(e)]}


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Reddit Data Processor - Scrape, embed, and insert Reddit data')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', help='Path to JSON file containing scraped data')
    input_group.add_argument('--scrape', action='store_true', help='Scrape data directly from Reddit')
    
    # Scraping options
    parser.add_argument('--subreddit', '-s', default='AskReddit', help='Subreddit to scrape (default: AskReddit)')
    parser.add_argument('--max-posts', '-p', type=int, default=10, help='Maximum posts to scrape (default: 10)')
    parser.add_argument('--max-comments', '-c', type=int, default=20, help='Maximum comments per post (default: 20)')
    
    # Processing options
    parser.add_argument('--batch-size', '-b', type=int, default=50, help='Batch size for processing (default: 50)')
    
    args = parser.parse_args()
    
    processor = None
    try:
        processor = RedditProcessor()
        
        if args.scrape:
            # Scrape and process directly
            result = processor.scrape_and_process(
                subreddit=args.subreddit,
                max_posts=args.max_posts,
                max_comments_per_post=args.max_comments
            )
            print(f"Scraped and processed: {result['posts']} posts, {result['comments']} comments")
            
        else:
            # Load and process from file
            result = processor.load_and_process(args.input, args.batch_size)
            print(f"Loaded and processed: {result['posts']} posts, {result['comments']} comments")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        if processor:
            processor.close()
    
    return 0


if __name__ == "__main__":
    exit(main())

__all__ = ['RedditProcessor']