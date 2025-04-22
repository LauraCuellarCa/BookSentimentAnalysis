import re
import os
import json
import numpy as np
# Set matplotlib to use non-interactive backend before other imports
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from collections import Counter
import threading

# Ensure necessary NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class BookAnalyzer:
    """Simplified book analyzer that extracts sentiment and generates visualizations"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Define emotion categories
        self.emotion_categories = [
            'joy', 'sadness', 'anger', 'fear', 'love', 
            'surprise', 'confusion', 'anticipation', 'trust', 'disgust'
        ]

    def analyze_book(self, title, text, progress_callback=None):
        """
        Analyze a book and generate a profile with sentiment and thematic info
        
        Args:
            title (str): Book title
            text (str): Full book text
            progress_callback (callable): Function to report progress
            
        Returns:
            dict: Book profile containing sentiment and topic information
        """
        # Clean text (remove headers, footers if present)
        text = self._clean_text(text)
        
        if progress_callback:
            progress_callback("Text cleaned, segmenting chapters...", 25)
        
        # Segment into chapters
        chapters = self._segment_chapters(text)
        
        if progress_callback:
            progress_callback("Analyzing sentiment and emotions...", 40)
        
        # Process sentiment by chapter
        sentiment_data = self._analyze_sentiment_by_chapter(chapters)
        
        if progress_callback:
            progress_callback("Extracting topics...", 60)
        
        # Extract topics
        topics, topic_distribution = self._extract_topics(chapters)
        
        if progress_callback:
            progress_callback("Identifying characters...", 80)
        
        # Find main characters and their associated emotions
        character_emotions = self._extract_character_emotions(text)
        
        # Compile book profile
        profile = {
            "title": title,
            "num_chapters": len(chapters),
            "sentiment_trajectory": sentiment_data,
            "topics": topics,
            "chapter_topic_distribution": topic_distribution.tolist() if isinstance(topic_distribution, np.ndarray) else topic_distribution,
            "character_emotions": character_emotions
        }
        
        return profile
    
    def _clean_text(self, text):
        """Remove common headers/footers from text"""
        # Remove Project Gutenberg header/footer if present
        header_pattern = r'Project Gutenberg.*?\*\*\* START OF THIS PROJECT GUTENBERG.*?\*\*\*'
        footer_pattern = r'\*\*\* END OF THIS PROJECT GUTENBERG.*'
        
        text = re.sub(header_pattern, '', text, flags=re.DOTALL)
        text = re.sub(footer_pattern, '', text, flags=re.DOTALL)
        
        return text.strip()
    
    def _segment_chapters(self, text):
        """Segment book into chapters based on common patterns"""
        # Match common chapter patterns
        chapter_patterns = [
            r'Chapter \d+', r'CHAPTER \d+',
            r'Chapter [IVXLCDM]+', r'CHAPTER [IVXLCDM]+'
        ]
        
        pattern = '|'.join(chapter_patterns)
        chapters = re.split(pattern, text)
        
        # Remove empty chapters and trim whitespace
        chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
        
        # If no chapters found or very few, try to split by newlines
        if len(chapters) <= 1:
            paragraphs = text.split('\n\n')
            chunks = []
            chunk_size = max(1, len(paragraphs) // 10)  # Aim for about 10 chunks
            
            for i in range(0, len(paragraphs), chunk_size):
                chunks.append('\n\n'.join(paragraphs[i:i+chunk_size]))
            
            return chunks
        
        return chapters
    
    def _analyze_sentiment_by_chapter(self, chapters):
        """Extract sentiment and emotions for each chapter"""
        polarity_by_chapter = []
        emotions_by_chapter = {emotion: [] for emotion in self.emotion_categories}
        
        for chapter in chapters:
            # Extract polarity using TextBlob
            blob = TextBlob(chapter)
            polarity = blob.sentiment.polarity
            polarity_by_chapter.append(polarity)
            
            # Simplified emotion detection
            emotion_scores = self._detect_emotions(chapter)
            for emotion, score in emotion_scores.items():
                emotions_by_chapter[emotion].append(score)
        
        # Apply smoothing for better visualization
        smoothed_polarity = self._smooth_data(polarity_by_chapter)
        smoothed_emotions = {
            emotion: self._smooth_data(scores)
            for emotion, scores in emotions_by_chapter.items()
        }
        
        return {
            "raw": {
                "polarity": polarity_by_chapter,
                "emotions": emotions_by_chapter
            },
            "smoothed": {
                "polarity": smoothed_polarity,
                "emotions": smoothed_emotions
            }
        }
    
    def _detect_emotions(self, text):
        """
        Simplified emotion detection based on keyword matching.
        In a full implementation, this would use a more sophisticated NLP model.
        """
        emotion_lexicon = {
            'joy': ['happy', 'joy', 'delight', 'excited', 'glad', 'pleasure'],
            'sadness': ['sad', 'sorrow', 'grief', 'depressed', 'unhappy', 'miserable'],
            'anger': ['angry', 'rage', 'fury', 'annoyed', 'mad', 'frustration'],
            'fear': ['afraid', 'fear', 'terror', 'panic', 'dread', 'scared'],
            'love': ['love', 'adore', 'affection', 'fond', 'passion', 'tender'],
            'surprise': ['surprise', 'shock', 'astonish', 'amaze', 'startled'],
            'confusion': ['confused', 'puzzled', 'perplexed', 'bewildered'],
            'anticipation': ['anticipate', 'expect', 'hope', 'await', 'looking forward'],
            'trust': ['trust', 'belief', 'faith', 'confidence', 'rely'],
            'disgust': ['disgust', 'revulsion', 'repel', 'gross', 'nausea']
        }
        
        # Tokenize text
        words = text.lower().split()
        
        # Count emotion keywords
        emotion_counts = {emotion: 0 for emotion in self.emotion_categories}
        
        for word in words:
            for emotion, keywords in emotion_lexicon.items():
                if any(keyword in word for keyword in keywords):
                    emotion_counts[emotion] += 1
        
        # Normalize by text length
        total_words = len(words) or 1  # Avoid division by zero
        emotion_scores = {
            emotion: min(1.0, count / (total_words * 0.05))  # Cap at 1.0
            for emotion, count in emotion_counts.items()
        }
        
        return emotion_scores
    
    def _smooth_data(self, data, window=3):
        """Apply smoothing to data for better visualization"""
        if len(data) <= window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            # Calculate window boundaries
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            
            # Calculate average
            window_avg = sum(data[start:end]) / (end - start)
            smoothed.append(window_avg)
        
        return smoothed
    
    def _extract_topics(self, chapters, num_topics=5):
        """Extract main topics from chapters using TF-IDF and LDA"""
        # Preprocess chapters
        processed_chapters = []
        for chapter in chapters:
            # Lowercase and remove punctuation
            chapter = re.sub(r'[^\w\s]', '', chapter.lower())
            
            # Tokenize and lemmatize
            tokens = word_tokenize(chapter)
            filtered_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in self.stop_words
            ]
            
            processed_chapters.append(' '.join(filtered_tokens))
        
        # Apply TF-IDF
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        
        # Handle empty chapters
        if not processed_chapters or all(not chapter for chapter in processed_chapters):
            dummy_topics = {f"topic_{i}": ["no", "content", "found"] for i in range(num_topics)}
            dummy_dist = [[1/num_topics] * num_topics] * len(chapters)
            return dummy_topics, dummy_dist
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_chapters)
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply LDA for topic modeling
            lda = LatentDirichletAllocation(
                n_components=num_topics, 
                max_iter=10, 
                learning_method='online',
                random_state=42
            )
            
            topic_distribution = lda.fit_transform(tfidf_matrix)
            
            # Extract top words for each topic
            topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-10-1:-1]  # Get top 10 words
                top_words = [feature_names[i] for i in top_words_idx]
                topics[f"topic_{topic_idx}"] = top_words
            
            return topics, topic_distribution
        except Exception as e:
            print(f"Error in topic extraction: {str(e)}")
            # Return dummy data on error
            dummy_topics = {f"topic_{i}": ["analysis", "failed", "try", "again"] for i in range(num_topics)}
            dummy_dist = [[1/num_topics] * num_topics] * len(chapters)
            return dummy_topics, dummy_dist
    
    def _extract_character_emotions(self, text):
        """
        Extract main characters and their associated emotions.
        This is a simplified version that looks for capitalized words as potential character names.
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Find potential character names (capitalized words not at the start of sentences)
        name_pattern = r'(?<!\. )[A-Z][a-z]{2,15}'
        potential_names = []
        
        for sentence in sentences:
            matches = re.findall(name_pattern, sentence)
            potential_names.extend(matches)
        
        # Count occurrences and find top characters
        name_counts = Counter(potential_names)
        top_characters = [name for name, count in name_counts.most_common(10) if count > 5]
        
        # Associate emotions with characters
        character_emotions = {}
        
        for character in top_characters:
            # Find sentences containing this character
            character_sentences = [s for s in sentences if character in s]
            
            if character_sentences:
                # Join sentences for context
                context = ' '.join(character_sentences)
                
                # Detect emotions in this context
                emotions = self._detect_emotions(context)
                character_emotions[character] = emotions
        
        return character_emotions
    
    def generate_visualizations(self, profile, output_dir):
        """Generate visualizations from book profile and save to output directory"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot sentiment trajectory
        self._plot_sentiment_trajectory(profile, output_dir)
        
        # Plot character emotions
        self._plot_character_emotions(profile, output_dir)
        
        # Plot topic heatmap
        self._plot_topic_heatmap(profile, output_dir)
    
    def _plot_sentiment_trajectory(self, profile, output_dir):
        """Plot sentiment trajectory (polarity and emotions) over chapters"""
        # Get data
        trajectory = profile["sentiment_trajectory"]["smoothed"]
        polarity = trajectory["polarity"]
        emotions = trajectory["emotions"]
        
        # Plot polarity
        plt.figure(figsize=(10, 5))
        plt.plot(polarity, 'b-', linewidth=2)
        plt.title(f"Sentiment Polarity: {profile['title']}")
        plt.xlabel("Chapter")
        plt.ylabel("Polarity (-1 to 1)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "polarity_trajectory.png"))
        plt.close()
        
        # Plot top emotions
        plt.figure(figsize=(12, 6))
        
        # Calculate average intensity for each emotion
        avg_intensity = {e: np.mean(v) for e, v in emotions.items()}
        top_emotions = sorted(avg_intensity.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for emotion, values in [(e, emotions[e]) for e, _ in top_emotions]:
            plt.plot(values, label=emotion.capitalize())
        
        plt.title(f"Emotional Trajectory: {profile['title']}")
        plt.xlabel("Chapter")
        plt.ylabel("Emotion Intensity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "emotion_trajectory.png"))
        plt.close()
    
    def _plot_character_emotions(self, profile, output_dir):
        """Plot emotions associated with main characters"""
        character_emotions = profile["character_emotions"]
        
        if not character_emotions:
            return
        
        # Get top 5 characters by total emotion intensity
        top_chars = sorted(
            character_emotions.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )[:5]
        
        if not top_chars:
            return
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set width of bars
        bar_width = 0.15
        
        # Set positions on X axis
        emotions = list(next(iter(character_emotions.values())).keys())
        r = np.arange(len(emotions))
        
        # Create bars
        for i, (char, char_emotions) in enumerate(top_chars):
            emotion_values = [char_emotions[e] for e in emotions]
            ax.bar(r + i * bar_width, emotion_values, width=bar_width, label=char)
        
        # Add labels
        plt.xlabel('Emotions')
        plt.ylabel('Intensity')
        plt.title(f"Character Emotions: {profile['title']}")
        plt.xticks([r + bar_width * 2 for r in range(len(emotions))], 
                   [e.capitalize() for e in emotions], 
                   rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "character_emotions.png"))
        plt.close()
    
    def _plot_topic_heatmap(self, profile, output_dir):
        """Plot a heatmap of topics by chapter"""
        topic_dist = np.array(profile["chapter_topic_distribution"])
        topics = profile["topics"]
        
        if topic_dist.size == 0:
            return
        
        # Number of chapters and topics
        n_chapters, n_topics = topic_dist.shape
        
        # Create labels for topics
        topic_labels = []
        for i in range(n_topics):
            top_words = topics.get(f"topic_{i}", ["no", "topic", "found"])[:3]  # Get top 3 words
            topic_labels.append(f"Topic {i}: {', '.join(top_words)}")
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(topic_dist, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Topic Probability')
        
        # Add labels
        plt.yticks(range(n_chapters), [f"Ch {i+1}" for i in range(n_chapters)])
        plt.xticks(range(n_topics), topic_labels, rotation=45, ha='right')
        
        plt.title(f"Topic Distribution: {profile['title']}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "topic_heatmap.png"))
        plt.close()

# Simple helper function to load books (can be expanded for PDF support)
def load_book(file_path):
    """Load a book from file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read() 