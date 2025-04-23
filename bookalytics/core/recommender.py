import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

class BookRecommender:
    def __init__(self, book_profiles):
        """
        Initialize the recommender with a list of book profiles.
        
        Args:
            book_profiles (list): List of book profile dictionaries
        """
        self.book_profiles = book_profiles  # List of book_profile dicts
        self.vectorized_books = self._vectorize_books(book_profiles)
        logger.info(f"Initialized BookRecommender with {len(book_profiles)} books")

    def _vectorize_books(self, profiles):
        """
        Create a vector representation of each book using emotion profile + trajectory.
        
        Args:
            profiles (list): List of book profile dictionaries
            
        Returns:
            tuple: (numpy array of vectors, list of metadata dictionaries)
        """
        if not profiles:
            return np.array([]), []
            
        vectors = []
        meta = []
        
        # Get all possible emotion keys from all books
        all_emotions = set()
        for book in profiles:
            all_emotions.update(book.get("overall_emotions", {}).keys())
        all_emotions = sorted(list(all_emotions))
        
        for book in profiles:
            # Create emotion vector
            emotion_vector = [book.get("overall_emotions", {}).get(e, 0) for e in all_emotions]
            
            # Extract trajectory data
            trajectory_vector = 0
            if "sentiment_trajectory" in book and "smoothed" in book["sentiment_trajectory"]:
                emotions_data = book["sentiment_trajectory"]["smoothed"].get("emotions", {})
                if emotions_data:
                    trajectory_vector = np.mean([
                        np.mean(v) if isinstance(v, list) else v
                        for v in emotions_data.values()
                    ])
            
            # Combine vectors
            combined_vector = np.append(emotion_vector, trajectory_vector)
            vectors.append(combined_vector)
            
            # Store metadata
            genre = book.get("genre", "unknown")
            meta.append({"title": book["title"], "genre": genre})
        
        return np.array(vectors), meta

    def recommend_similar(self, book_title, top_k=3):
        """
        Recommend books most similar to the given title (hybrid filtering).
        
        Args:
            book_title (str): Title of the book to find recommendations for
            top_k (int): Number of recommendations to return
            
        Returns:
            list: List of recommended book titles
        """
        vectors, meta = self.vectorized_books
        if vectors.size == 0:
            return []
            
        titles = [m['title'] for m in meta]
        if book_title not in titles:
            logger.warning(f"Book title '{book_title}' not found in dataset")
            return []

        idx = titles.index(book_title)
        sims = cosine_similarity([vectors[idx]], vectors)[0]
        sim_scores = sorted([(i, score) for i, score in enumerate(sims) if i != idx], 
                           key=lambda x: x[1], reverse=True)

        return [(meta[i]["title"], float(score)) for i, score in sim_scores[:top_k]]

    def recommend_by_genre_and_emotion(self, genre, eps=0.5, min_samples=2):
        """
        Recommend books within a genre based on emotional clustering.
        
        Args:
            genre (str): Genre to filter by
            eps (float): DBSCAN epsilon parameter
            min_samples (int): DBSCAN min_samples parameter
            
        Returns:
            dict: Dictionary of clusters with book lists
        """
        vectors, meta = self.vectorized_books
        if vectors.size == 0:
            return {}
            
        filtered = [(v, m) for v, m in zip(vectors, meta) if m["genre"].lower() == genre.lower()]
        if not filtered:
            logger.warning(f"No books found with genre '{genre}'")
            return {}

        filtered_vectors = np.array([f[0] for f in filtered])
        filtered_meta = [f[1] for f in filtered]

        # Only proceed with clustering if we have enough samples
        if len(filtered_vectors) < min_samples:
            logger.warning(f"Not enough books ({len(filtered_vectors)}) for clustering with min_samples={min_samples}")
            return {"0": filtered_meta}  # Return all as one cluster

        # Standardize the data
        scaler = StandardScaler()
        scaled = scaler.fit_transform(filtered_vectors)

        # Apply clustering
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled)
            labels = clustering.labels_
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            return {"0": filtered_meta}  # Return all as one cluster

        clusters = {}
        for label, book in zip(labels, filtered_meta):
            if label == -1:
                continue  # skip outliers
            # Convert NumPy int64 to regular Python string
            cluster_key = str(int(label))
            clusters.setdefault(cluster_key, []).append(book)

        return clusters

    def recommend_by_emotion(self, target_emotion, threshold=0.1):
        """
        Recommend books where a dominant emotional pattern matches a target.
        
        Args:
            target_emotion (str): Target emotion to look for
            threshold (float): Minimum score threshold
            
        Returns:
            list: List of (title, score) tuples
        """
        matches = []
        for book in self.book_profiles:
            emotion_score = book.get("overall_emotions", {}).get(target_emotion, 0)
            if emotion_score >= threshold:
                matches.append((book["title"], float(emotion_score)))

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def get_recommendations_for_book(self, book_title):
        """
        Get comprehensive recommendations for a book.
        
        Args:
            book_title (str): Title of the book
            
        Returns:
            dict: Dictionary with different recommendation types
        """
        # Find the book in profiles
        book_idx = None
        for i, book in enumerate(self.book_profiles):
            if book.get("title") == book_title:
                book_idx = i
                break
                
        if book_idx is None:
            return {"error": "Book not found"}
            
        book = self.book_profiles[book_idx]
        
        # Get genre
        genre = book.get("genre", "unknown")
        
        # Get top emotions
        top_emotions = []
        if "overall_emotions" in book:
            top_emotions = sorted(
                book["overall_emotions"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
        
        recommendations = {
            "similar_books": self.recommend_similar(book_title, top_k=3),
            "genre_clusters": self.recommend_by_genre_and_emotion(genre, eps=1.5, min_samples=1),
            "emotional_matches": []
        }
        
        # Add emotional recommendations for top emotions
        for emotion, _ in top_emotions:
            emotional_recs = self.recommend_by_emotion(emotion, threshold=0.1)
            # Filter out the current book
            emotional_recs = [(title, score) for title, score in emotional_recs 
                             if title != book_title][:2]
            
            if emotional_recs:
                recommendations["emotional_matches"].append({
                    "emotion": emotion,
                    "matches": emotional_recs
                })
        
        # Convert any NumPy values to native Python types
        return self._to_json_serializable(recommendations)
        
    def _to_json_serializable(self, obj):
        """
        Convert NumPy types to Python native types for JSON serialization.
        
        Args:
            obj: Any Python object
            
        Returns:
            object: JSON serializable object
        """
        if isinstance(obj, dict):
            return {self._to_json_serializable(k): self._to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [self._to_json_serializable(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._to_json_serializable(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj 