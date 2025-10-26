#!/usr/bin/env python3
"""
Analyze UMAP Clusters for Comments
=================================

This script analyzes what topics are actually in your UMAP clusters for comments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import RedditDataManager
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.cluster import KMeans

def analyze_comment_clusters():
    """Analyze what topics are in your UMAP clusters for comments."""
    
    # Try to import UMAP
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return
    
    db = RedditDataManager()
    
    try:
        # Get comments
        comments = list(db.mongo.comments_collection.find().limit(200))  # More comments for better clustering
        
        if len(comments) < 10:
            print("Not enough comments for analysis")
            return
        
        # Extract embeddings and text
        embeddings = []
        bodies = []
        scores = []
        
        for comment in comments:
            if "embedding" in comment and comment["embedding"]:
                embeddings.append(comment["embedding"])
                bodies.append(comment.get("body", ""))
                scores.append(comment.get("score", 0))
        
        embeddings_array = np.array(embeddings)
        
        # Use optimized UMAP parameters
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=5,  # Tight clusters
            min_dist=0.05,  # Very tight
            metric='cosine',
            random_state=42
        )
        
        umap_embedding = reducer.fit_transform(embeddings_array)
        
        # Find optimal number of clusters
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = {}
        
        for k in [2, 3, 4, 5, 6, 7, 8, 10]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(umap_embedding)
            silhouette_avg = silhouette_score(umap_embedding, cluster_labels)
            silhouette_scores[k] = silhouette_avg
        
        # Find best K
        best_k = max(silhouette_scores.items(), key=lambda x: x[1])[0]
        best_silhouette = silhouette_scores[best_k]
        
        print(f"Best clustering: K={best_k} (silhouette={best_silhouette:.4f})")
        
        chosen_k = best_k
        
        # Get final cluster labels
        kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(umap_embedding)
        
        # Create simple cluster visualization
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(
            umap_embedding[:, 0], umap_embedding[:, 1],
            c=cluster_labels, cmap='tab10', alpha=0.7, s=50
        )
        
        plt.title(f'UMAP Clustering for Comments - {chosen_k} Clusters')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.colorbar(scatter, label='Cluster')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze each cluster
        
        for cluster_id in range(chosen_k):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_comments = [comments[i] for i in cluster_indices]
            
            print(f"\nCluster {cluster_id} ({len(cluster_comments)} comments):")
            
            # Extract keywords from this cluster
            all_text = " ".join([bodies[i] for i in cluster_indices])
            
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            
            # Remove common stop words
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'this', 'that', 'with', 'have', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'uber', 'lyft', 'driver', 'driving', 'ride', 'passenger', 'car', 'money', 'pay', 'hour', 'work'
            }
            
            filtered_words = [w for w in words if w not in stop_words]
            word_counts = Counter(filtered_words)
            
            # Calculate average score for this cluster
            cluster_scores = [scores[i] for i in cluster_indices]
            avg_score = np.mean(cluster_scores)
            
            # Try to identify the topic
            top_words = [word for word, count in word_counts.most_common(5)]
            print(f"  Topic: {' '.join(top_words[:3])} | Score: {avg_score:.1f} | Keywords: {[word for word, count in word_counts.most_common(5)]}")
        
        # Show cluster distribution
        cluster_counts = Counter(cluster_labels)
        print(f"\nCluster Distribution:")
        for cluster_id, count in sorted(cluster_counts.items()):
            percentage = (count / len(cluster_labels)) * 100
            print(f"  Cluster {cluster_id}: {count} comments ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error during comment cluster analysis: {e}")
    
    finally:
        db.mongo.close()

if __name__ == "__main__":
    analyze_comment_clusters()
