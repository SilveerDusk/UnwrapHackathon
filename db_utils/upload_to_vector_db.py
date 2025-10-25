from db_utils.reddit_processor import RedditProcessor
from post_utils.redditCaller import fetch_posts, fetch_comments, fetch_subreddit_posts

def main():
    """Main function with command line argument parsing"""
    processor = None
    try:
        processor = RedditProcessor()
        
        subreddit_name = "uberdrivers"
        number_of_posts = 200  # Example: Fetch 200 posts

        posts = fetch_subreddit_posts(subreddit_name, number_of_posts)
        print(f"Fetched {len(posts)} posts from subreddit '{subreddit_name}'")

        for post in posts:
            post_id = post["id"]
            comments = fetch_comments(post_id)
            print(f"Fetched {len(comments)} comments for post ID {post_id}")

            processed_post = processor.process_post(post)
            processor.db_manager.insert_post(processed_post)
            for comment in comments:
                processor.process_comment(comment, post_id)
                processor.db_manager.insert_comment(comment)
  
        print("Data insertion complete.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        if processor:
            processor.close()
    
    return 0


if __name__ == "__main__":
    main()