import os
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv
import logging
from reddit_scraper import RedditScraper

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBConnection:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.posts_collection: Optional[Collection] = None
        self.comments_collection: Optional[Collection] = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            connection_string = os.getenv("MONGODB_URI")
            if not connection_string:
                raise ValueError("MONGODB_URI environment variable not set")
            
            self.client = MongoClient(connection_string)
            self.db = self.client.reddit
            self.posts_collection = self.db.posts
            self.comments_collection = self.db.comments
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

class RedditDataManager:
    def __init__(self):
        self.mongo = MongoDBConnection()
    
    def insert_post(self, post_data: Dict) -> str:
        """Insert a single post into the database"""
        try:
            # Add metadata
            post_data['created_at'] = datetime.fromtimestamp(post_data['created_utc'])
            post_data['inserted_at'] = datetime.now(timezone.utc)
            
            result = self.mongo.posts_collection.insert_one(post_data)
            logger.info(f"Inserted post {post_data['id']} with ID {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert post {post_data.get('id', 'unknown')}: {e}")
            raise
    
    def insert_posts_batch(self, posts_data: List[Dict]) -> List[str]:
        """Insert multiple posts in a batch"""
        try:
            # Add metadata to all posts
            for post in posts_data:
                post['created_at'] = datetime.fromtimestamp(post['created_utc'])
                post['inserted_at'] = datetime.now(timezone.utc)
            
            result = self.mongo.posts_collection.insert_many(posts_data)
            logger.info(f"Inserted {len(result.inserted_ids)} posts")
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            logger.error(f"Failed to insert posts batch: {e}")
            raise
    
    def insert_comment(self, comment_data: Dict) -> str:
        """Insert a single comment into the database"""
        try:
            # Add metadata
            comment_data['created_at'] = datetime.fromtimestamp(comment_data['created_utc'])
            comment_data['inserted_at'] = datetime.now(timezone.utc)
            
            result = self.mongo.comments_collection.insert_one(comment_data)
            logger.info(f"Inserted comment {comment_data['id']} with ID {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert comment {comment_data.get('id', 'unknown')}: {e}")
            raise
    
    def insert_comments_batch(self, comments_data: List[Dict]) -> List[str]:
        """Insert multiple comments in a batch"""
        try:
            # Add metadata to all comments
            for comment in comments_data:
                comment['created_at'] = datetime.fromtimestamp(comment['created_utc'])
                comment['inserted_at'] = datetime.now(timezone.utc)
            
            result = self.mongo.comments_collection.insert_many(comments_data)
            logger.info(f"Inserted {len(result.inserted_ids)} comments")
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            logger.error(f"Failed to insert comments batch: {e}")
            raise
    
    def get_posts_by_date_range(self, subreddit: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get posts within a date range, optionally filtering out bot posts"""
        try:
            query = {
                'subreddit': subreddit,
                'created_at': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            posts = list(self.mongo.posts_collection.find(query).sort('created_at', -1))
            logger.info(f"Retrieved {len(posts)} posts for {subreddit} between {start_date} and {end_date}")
            return posts
        except Exception as e:
            logger.error(f"Failed to get posts by date range: {e}")
            raise
    
    def get_comments_by_date_range(self, subreddit: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get comments within a date range, optionally filtering out bot comments"""
        try:
            query = {
                'subreddit': subreddit,
                'created_at': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            comments = list(self.mongo.comments_collection.find(query).sort('created_at', -1))
            logger.info(f"Retrieved {len(comments)} comments for {subreddit} between {start_date} and {end_date}")
            return comments
        except Exception as e:
            logger.error(f"Failed to get comments by date range: {e}")
            raise
    
    def get_post_with_comments(self, post_id: str) -> Dict:
        """Get a post with all its comments for comprehensive analysis"""
        try:
            # Get the post
            post = self.mongo.posts_collection.find_one({"id": post_id})
            if not post:
                return None
            
            # Get all comments for this post
            comments = list(self.mongo.comments_collection.find({"post_id": post_id}))
            
            return {
                "post": post,
                "comments": comments,
                "total_comments": len(comments),
                "post_id": post_id
            }
        except Exception as e:
            logger.error(f"Failed to get post with comments for {post_id}: {e}")
            raise
    
    # Use in the future for LLM analysis to get insights from the post and comments
    def get_combined_content_for_analysis(self, post_id: str) -> str:
        """Get combined post + comments content for LLM analysis"""
        try:
            post_data = self.get_post_with_comments(post_id)
            if not post_data:
                return ""
            
            post = post_data["post"]
            comments = post_data["comments"]
            
            # Combine post content
            post_content = f"POST: {post.get('title', '')} {post.get('selftext', '')}"
            
            # Add top comments (by score)
            sorted_comments = sorted(comments, key=lambda x: x.get('score', 0), reverse=True)
            top_comments = sorted_comments[:10]  # Top 10 comments
            
            comment_content = "\n".join([f"COMMENT: {comment.get('body', '')}" for comment in top_comments])
            
            return f"{post_content}\n\n{comment_content}"
        except Exception as e:
            logger.error(f"Failed to get combined content for {post_id}: {e}")
            return ""
    
    def search_posts_and_comments(self, query: str, subreddit: str, limit: int = 5) -> Dict:
        """
        Search posts first, then get comments for those specific posts.
        This ensures comments are related to the posts in the results.
        """
        try:
            scraper = RedditScraper()
            query_embedding = scraper.generate_embedding(query)
            
            # Search posts first
            posts_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "posts_vector_idx",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit,
                        "filter": {"subreddit": subreddit}
                    }
                },
                {
                    "$project": {
                        "id": 1, "title": 1, "score": 1, "created_at": 1, "author": 1,
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            posts_results = list(self.mongo.posts_collection.aggregate(posts_pipeline))
            
            # Get post IDs from the results
            post_ids = [post["id"] for post in posts_results]
            
            # Get comments for these specific posts
            comments_results = []
            if post_ids:
                comments_pipeline = [
                    {
                        "$match": {
                            "post_id": {"$in": post_ids},
                            "subreddit": subreddit
                        }
                    },
                    {
                        "$vectorSearch": {
                            "index": "comments_vector_idx",
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": limit * 20,  # More candidates since we're filtering by post_id
                            "limit": limit * 3  # More comments since we're filtering by post_id
                        }
                    },
                    {
                        "$project": {
                            "id": 1, "body": 1, "post_id": 1, "score": 1, "created_at": 1, "author": 1,
                            "similarity_score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                
                comments_results = list(self.mongo.comments_collection.aggregate(comments_pipeline))
            
            return {
                "query": query,
                "posts": posts_results,
                "comments": comments_results,
                "total_posts": len(posts_results),
                "total_comments": len(comments_results),
                "related_posts": post_ids
            }
        except Exception as e:
            logger.error(f"Failed to search posts and comments: {e}")
            return {"posts": [], "comments": [], "total_posts": 0, "total_comments": 0, "related_posts": []}
    
    def search_posts_with_top_comments(self, query: str, subreddit: str, limit: int = 5, comments_per_post: int = 3) -> Dict:
        """
        Search posts and return each post with its top comments (by score).
        This is more useful for topic analysis.
        """
        try:
            # Generate query embedding
            from reddit_scraper import RedditScraper
            scraper = RedditScraper()
            query_embedding = scraper.generate_embedding(query)
            
            # Search posts first
            posts_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "posts_vector_idx",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit,
                        "filter": {"subreddit": subreddit}
                    }
                },
                {
                    "$project": {
                        "id": 1, "title": 1, "score": 1, "created_at": 1, "author": 1,
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            posts_results = list(self.mongo.posts_collection.aggregate(posts_pipeline))
            
            # For each post, get its top comments
            posts_with_comments = []
            for post in posts_results:
                post_id = post["id"]
                
                # Get top comments for this post (by score, not vector similarity)
                top_comments = list(self.mongo.comments_collection.find(
                    {"post_id": post_id, "subreddit": subreddit}
                ).sort("score", -1).limit(comments_per_post))
                
                posts_with_comments.append({
                    "post": post,
                    "comments": top_comments,
                    "comment_count": len(top_comments)
                })
            
            return {
                "query": query,
                "posts_with_comments": posts_with_comments,
                "total_posts": len(posts_results),
                "total_comments": sum(len(p["comments"]) for p in posts_with_comments)
            }
        except Exception as e:
            logger.error(f"Failed to search posts with top comments: {e}")
            return {"posts_with_comments": [], "total_posts": 0, "total_comments": 0}
    
    # Use in the future for LLM analysis to get insights from the post and comments - CHECK LATER
    
    # def get_topic_insights(self, subreddit: str, start_date: datetime, end_date: datetime,
    #                       include_bots: bool = True) -> Dict:
    #     """Get aggregated insights for a subreddit in a date range"""
    #     try:
    #         posts = self.get_posts_by_date_range(subreddit, start_date, end_date, include_bots)
    #         comments = self.get_comments_by_date_range(subreddit, start_date, end_date, include_bots)
            
    #         # Calculate basic statistics
    #         total_posts = len(posts)
    #         total_comments = len(comments)
    #         total_score = sum(post.get('score', 0) for post in posts)
    #         avg_score = total_score / total_posts if total_posts > 0 else 0
            
    #         # Count bot vs human content
    #         bot_posts = sum(1 for post in posts if post.get('is_bot', False))
    #         bot_comments = sum(1 for comment in comments if comment.get('is_bot', False))
            
    #         insights = {
    #             'subreddit': subreddit,
    #             'date_range': {
    #                 'start': start_date.isoformat(),
    #                 'end': end_date.isoformat()
    #             },
    #             'statistics': {
    #                 'total_posts': total_posts,
    #                 'total_comments': total_comments,
    #                 'total_score': total_score,
    #                 'avg_score': avg_score,
    #                 'bot_posts': bot_posts,
    #                 'bot_comments': bot_comments,
    #                 'human_posts': total_posts - bot_posts,
    #                 'human_comments': total_comments - bot_comments
    #             },
    #             'posts': posts,
    #             'comments': comments
    #         }
            
    #         logger.info(f"Generated insights for {subreddit}: {total_posts} posts, {total_comments} comments")
    #         return insights
    #     except Exception as e:
    #         logger.error(f"Failed to get topic insights: {e}")
    #         raise

# Example usage and testing
if __name__ == "__main__":
    # Test the database connection
    try:
        db_manager = RedditDataManager()
        print("Database connection successful!")
        
        # Test inserting a sample post
        sample_post = {
            'id': 'test123',
            'title': 'Test Post',
            'subreddit': 'test',
            'created_utc': 1640995200,  # 2022-01-01
            'score': 10,
            'num_comments': 5,
            'author': 'testuser',
            'embedding': [0.1] * 384  # Dummy embedding
        }
        
        # Uncomment to test insertion
        # post_id = db_manager.insert_post(sample_post)
        # print(f"Inserted post with ID: {post_id}")
        
    except Exception as e:
        print(f"Database test failed: {e}")
    finally:
        if 'db_manager' in locals():
            db_manager.mongo.close()
