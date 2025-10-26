import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from post_utils.redditCaller import RedditCaller
from tqdm import tqdm

def main():
    """Main function - complete pipeline for fetching, processing, and storing Reddit data"""
    caller = None
    try:
        print("Starting Reddit data pipeline...")
        caller = RedditCaller()
        
        subreddit_name = "SouthwestAirlines"
        number_of_posts = 200  # Example: Fetch 200 posts

        # Step 1: Fetch posts
        print(f"Fetching {number_of_posts} posts from r/{subreddit_name}...")
        posts = caller.fetch_subreddit_posts(subreddit_name, number_of_posts)
        print(f"Fetched {len(posts)} posts")

        # Step 2: Process and store posts and comments
        stored_posts = 0
        stored_comments = 0
        errors = []

        for post in tqdm(posts):
            try:
                #print(f"Processing post {i}/{len(posts)}: {post['title'][:50]}...")
                
                # Process post with embedding
                processed_post = caller.process_post(post)
                
                # Store post in database
                caller.db_manager.insert_post(processed_post)
                stored_posts += 1
                #print(f"Stored post: {post['id']}")
                
                # Step 3: Fetch comments for this post
                #print(f"Fetching comments for post {post['id']}...")
                comments = caller.fetch_comments(post['id'])
                #print(f"Found {len(comments)} comments")
                
                # Step 4: Process and store comments
                for comment in comments:
                    try:
                        # Process comment with embedding
                        processed_comment = caller.process_comment(comment)
                        
                        # Store comment in database
                        caller.db_manager.insert_comment(processed_comment)
                        stored_comments += 1
                        
                    except Exception as e:
                        error_msg = f"Failed to process comment {comment.get('id', 'unknown')}: {e}"
                        print(f"{error_msg}")
                        errors.append(error_msg)
                
                # Rate limiting - be respectful to Reddit's API
                import time
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Failed to process post {post.get('id', 'unknown')}: {e}"
                print(f"{error_msg}")
                errors.append(error_msg)
        
        # Final summary
        print(f"\nPipeline Complete!")
        print(f"Results:")
        print(f"  - Posts stored: {stored_posts}")
        print(f"  - Comments stored: {stored_comments}")
        print(f"  - Errors: {len(errors)}")
        
        if errors:
            print(f"\nErrors encountered:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        if caller:
            caller.close()
    
    return 0


if __name__ == "__main__":
    main()