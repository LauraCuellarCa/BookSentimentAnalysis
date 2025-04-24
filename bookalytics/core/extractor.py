import numpy as np
import pandas as pd
from collections import Counter
import re
import os
import json
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.ndimage import gaussian_filter1d
import difflib
import logging

from .model_cache import model_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sent_tokenize(text):
    # Basic sentence splitter using punctuation
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

class BookFeatureExtractor:
    def __init__(self):
        """Initialize the BookFeatureExtractor with NLP tools and models."""
        logger.info("Initializing BookFeatureExtractor")
        
        # Load SpaCy model from cache
        self.nlp = model_cache.get_spacy_model("en_core_web_sm")
        
        # Set up NLTK resources
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Set up emotion analysis
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
            "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        
        # Initialize emotion lexicon (simplified for demo)
        self.emotion_lexicon = {}
        
        # Get device from cache
        self.device = model_cache.device
        logger.info(f"Using device: {self.device}")
        
        # Load BERT model for emotion classification from cache
        self.model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
        self.tokenizer = model_cache.get_tokenizer(self.model_name)
        self.model = model_cache.get_transformer_model(self.model_name)
        
        if self.tokenizer and self.model:
            logger.info("BERT model loaded successfully from cache")
        else:
            logger.error("Failed to load BERT model from cache")
    
    def clean_gutenberg_text(self, text):
        """Remove Project Gutenberg metadata."""
        # Pattern to match Gutenberg header and footer
        start_markers = [
            "The Project Gutenberg EBook of",
            "Project Gutenberg's",
            "This eBook is for the use of anyone"
        ]
        end_markers = [
            "End of the Project Gutenberg EBook",
            "End of Project Gutenberg's",
            "*** END OF THIS PROJECT GUTENBERG EBOOK"
        ]
        
        # Find the start of the actual content
        start_pos = 0
        for marker in start_markers:
            if marker in text:
                marker_pos = text.find(marker)
                # Find the next paragraph break after the marker
                paragraph_break = text.find("\n\n", marker_pos)
                if paragraph_break > 0:
                    start_pos = max(start_pos, paragraph_break + 2)
        
        # Find the end of the actual content
        end_pos = len(text)
        for marker in end_markers:
            if marker in text:
                marker_pos = text.find(marker)
                # Find the last paragraph break before the marker
                paragraph_break = text.rfind("\n\n", 0, marker_pos)
                if paragraph_break > 0:
                    end_pos = min(end_pos, paragraph_break)
        
        # Extract the content
        if start_pos < end_pos:
            return text[start_pos:end_pos]
        return text

    def preprocess_text(self, text):
        """Preprocess text into tokens and sentences."""
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Process each sentence
        all_tokens = []
        for sentence in sentences:
            # Lowercase
            sentence = sentence.lower()
            # Remove non-alphabetic characters
            sentence = re.sub(r'[^a-zA-Z\s]', ' ', sentence)
            # Tokenize
            tokens = word_tokenize(sentence)
            # Remove stopwords
            tokens = [t for t in tokens if t not in self.stop_words]
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            
            all_tokens.extend(tokens)
        
        return all_tokens, sentences

    def segment_by_chapter(self, text):
        """Segment book text into chapters."""
        # Map of written number words to detect in chapter headings
        number_words = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine", "thirty",
            "thirty-one", "thirty-two", "thirty-three", "thirty-four", "thirty-five", "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine", "forty",
            "forty-one", "forty-two", "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven", "forty-eight", "forty-nine", "fifty",
            "fifty-one", "fifty-two", "fifty-three", "fifty-four", "fifty-five", "fifty-six", "fifty-seven", "fifty-eight", "fifty-nine", "sixty",
            "sixty-one", "sixty-two", "sixty-three", "sixty-four", "sixty-five", "sixty-six", "sixty-seven", "sixty-eight", "sixty-nine", "seventy",
            "seventy-one", "seventy-two", "seventy-three", "seventy-four", "seventy-five", "seventy-six", "seventy-seven", "seventy-eight", "seventy-nine", "eighty",
            "eighty-one", "eighty-two", "eighty-three", "eighty-four", "eighty-five", "eighty-six", "eighty-seven", "eighty-eight", "eighty-nine", "ninety",
            "ninety-one", "ninety-two", "ninety-three", "ninety-four", "ninety-five", "ninety-six", "ninety-seven", "ninety-eight", "ninety-nine", "one hundred"
        ]
        
        # Create regex pattern for written numbers
        written_numbers_pattern = "|".join(number_words)
        
        # Match common chapter patterns
        chapter_patterns = [
            # Numeric chapters
            r'Chapter \d+', r'CHAPTER \d+',
            # Roman numeral chapters
            r'Chapter [IVXLCDM]+', r'CHAPTER [IVXLCDM]+',
            # Written-out number chapters
            fr'Chapter (?:{written_numbers_pattern})\b', 
            fr'CHAPTER (?:{written_numbers_pattern})\b',
            # Alternative formats
            r'Chapter\s+\d+\.', r'CHAPTER\s+\d+\.',
            # Simple numbered sections
            r'\n\d+\.\s+', r'\n[IVXLCDM]+\.\s+'
        ]
        
        pattern = '|'.join(chapter_patterns)
        try:
            chapters = re.split(pattern, text, flags=re.IGNORECASE)
            
            # Remove empty chapters and trim whitespace
            chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
            
            logger.info(f"Found {len(chapters)} chapters using regex patterns")
            
            # If no chapters found or only one large chapter, try alternate approach
            if len(chapters) <= 1 and len(text) > 10000:
                logger.info("Trying alternate chapter detection method")
                # Fallback: Split by double newlines (paragraph breaks) and look for chapter-like headers
                paragraphs = text.split('\n\n')
                chapter_starts = []
                
                for i, para in enumerate(paragraphs):
                    para_lower = para.lower().strip()
                    if (para_lower.startswith('chapter') or 
                        re.match(r'^\d+\.?\s', para_lower) or 
                        re.match(r'^[ivxlcdm]+\.?\s', para_lower)):
                        chapter_starts.append(i)
                
                if chapter_starts:
                    # Use these indices to split the text
                    new_chapters = []
                    for i in range(len(chapter_starts)):
                        start_idx = chapter_starts[i]
                        end_idx = chapter_starts[i+1] if i < len(chapter_starts)-1 else len(paragraphs)
                        chapter_text = '\n\n'.join(paragraphs[start_idx:end_idx])
                        new_chapters.append(chapter_text)
                    
                    if len(new_chapters) > 1:
                        chapters = new_chapters
                        logger.info(f"Found {len(chapters)} chapters using alternate method")
            
            return chapters
            
        except Exception as e:
            logger.error(f"Error in chapter detection: {e}")
            # Fallback: Split into roughly equal chunks if regex fails
            approx_chapter_size = 5000  # characters
            if len(text) > approx_chapter_size * 2:
                chunks = [text[i:i+approx_chapter_size] for i in range(0, len(text), approx_chapter_size)]
                logger.info(f"Used fallback chunking: {len(chunks)} chunks")
                return chunks
            else:
                return [text]

    def extract_lexicon_based_emotions(self, tokens):
        """Extract emotions based on lexicon matching."""
        emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        
        for token in tokens:
            if token in self.emotion_lexicon:
                for emotion in self.emotion_lexicon[token]:
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1
        
        # Normalize counts
        total = sum(emotion_counts.values())
        emotion_scores = emotion_counts.copy()
        
        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = emotion_scores[emotion] / total
        
        return {
            "counts": emotion_counts,
            "scores": emotion_scores
        }

    def chunk_sentences(self, sentences, chunk_size=5):
        """Yield successive chunks from list of sentences."""
        for i in range(0, len(sentences), chunk_size):
            yield sentences[i:i + chunk_size]

    def extract_bert_emotions(self, text, batch_size=4):
        """Use fine-tuned BERT to classify text into emotions."""
        if self.tokenizer is None or self.model is None:
            return {label: 0.0 for label in self.emotion_labels}
            
        sentences = sent_tokenize(text)
        all_probs = []
        
        # Process in batches
        for batch in self.chunk_sentences(sentences, chunk_size=5):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            
        if not all_probs:
            return {label: 0.0 for label in self.emotion_labels}
        
        all_probs_tensor = torch.cat(all_probs, dim=0)
        avg_probs = torch.mean(all_probs_tensor, dim=0).numpy()
        
        num_output_labels = avg_probs.shape[0]
        used_labels = self.emotion_labels[:num_output_labels]
        emotion_scores = {label: float(avg_probs[i]) for i, label in enumerate(used_labels)}
        
        return emotion_scores

    def extract_polarity_with_textblob(self, sentences):
        """Extract sentiment polarity using TextBlob."""
        polarity_scores = []
        
        for sentence in sentences:
            analysis = TextBlob(sentence)
            polarity_scores.append(analysis.sentiment.polarity)
        
        return polarity_scores

    def extract_character_emotions(self, text):
        """Extract emotions associated with main characters using BERT."""
        paragraphs = text.split('\n\n')
        mention_counts = {}
        character_contexts = {}
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) == 0 or len(paragraph) > 100000:
                continue
            
            try:
                doc = self.nlp(paragraph)
            except Exception as e:
                logger.warning(f"Skipped paragraph due to parsing error: {e}")
                continue
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    mention_counts[name] = mention_counts.get(name, 0) + 1
                    character_contexts.setdefault(name, []).append(paragraph)
        
        # Merge similar character names
        merged = {}
        for name in character_contexts:
            found = False
            for canon in merged:
                if difflib.SequenceMatcher(None, name, canon).ratio() > 0.85:
                    merged[canon].extend(character_contexts[name])
                    found = True
                    break
            if not found:
                merged[name] = character_contexts[name]
        
        # Get top 10 most mentioned characters
        top_characters = sorted(mention_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        character_emotions = {}
        for name, _ in top_characters:
            context_text = " ".join(character_contexts[name])
            if context_text.strip():
                emotion_scores = self.extract_bert_emotions(context_text)
                character_emotions[name] = emotion_scores
        
        return character_emotions

    def get_custom_stopwords_from_entities(self, text, nlp, top_n=5):
        """Extract common entity names as stopwords."""
        doc = nlp(text)
        people = [ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"]
        most_common = [name for name, _ in Counter(people).most_common(top_n)]
        return set(most_common)

    def extract_tfidf_features(self, chapters):
        """Extract TF-IDF features from book chapters."""
        # Get common character names dynamically
        full_text = " ".join(chapters)
        custom_stopwords = self.get_custom_stopwords_from_entities(full_text, self.nlp)
        
        processed_chapters = []
        for chapter in chapters:
            # Lowercase
            chapter = chapter.lower()
            # Remove non-alphabetic characters
            chapter = re.sub(r'[^a-zA-Z\s]', ' ', chapter)
            
            # Tokenize with spaCy
            tokens = [token for token in self.nlp(chapter)]
            
            # Filter: keep nouns and adjectives, remove stopwords and named entities
            filtered_tokens = [
                self.lemmatizer.lemmatize(token.text.lower())
                for token in tokens
                if token.pos_ in {"NOUN", "ADJ"}
                and token.text.lower() not in self.stop_words
                and token.ent_type_ != "PERSON"
                and token.text.lower() not in custom_stopwords
            ]
            
            cleaned = " ".join(filtered_tokens)
            processed_chapters.append(cleaned)
        
        # Apply CountVectorizer with bigram support
        vectorizer = CountVectorizer(
            max_df=0.8,
            min_df=5,
            stop_words='english',
            ngram_range=(1, 2)  # unigrams and bigrams
        )
        tfidf_matrix = vectorizer.fit_transform(processed_chapters)
        
        return tfidf_matrix, vectorizer.get_feature_names_out()

    def extract_topics_with_lda(self, tfidf_matrix, feature_names, num_topics=10):
        """Extract topics using Latent Dirichlet Allocation."""
        # Create LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        
        # Fit model
        doc_topic_dist = lda.fit_transform(tfidf_matrix)
        
        # Extract topics
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f"topic_{topic_idx}"] = top_words
        
        return topics, doc_topic_dist

    def create_sentiment_trajectory(self, chapters):
        """Create a trajectory of sentiment through chapters."""
        polarity_trajectory = []
        emotion_trajectories = {emotion: [] for emotion in self.emotion_labels}
        
        for chapter in chapters:
            # Skip very short chapters
            if len(chapter.split()) < 50:
                continue
                
            # Extract polarity
            _, sentences = self.preprocess_text(chapter)
            polarity_scores = self.extract_polarity_with_textblob(sentences)
            avg_polarity = np.mean(polarity_scores) if polarity_scores else 0
            polarity_trajectory.append(avg_polarity)
            
            # Extract emotion scores
            emotion_scores = self.extract_bert_emotions(chapter)
            for emotion in self.emotion_labels:
                emotion_trajectories[emotion].append(
                    emotion_scores.get(emotion, 0)
                )
        
        return {
            "polarity": polarity_trajectory,
            "emotions": emotion_trajectories
        }

    def apply_gaussian_smoothing(self, trajectory, sigma=1):
        """Apply Gaussian smoothing to trajectories."""
        polarity = trajectory["polarity"]
        emotions = trajectory["emotions"]
        
        # Smooth polarity
        smoothed_polarity = gaussian_filter1d(polarity, sigma=sigma)
        
        # Smooth emotions
        smoothed_emotions = {}
        for emotion, values in emotions.items():
            if values:  # Only smooth if we have values
                smoothed_emotions[emotion] = gaussian_filter1d(values, sigma=sigma)
            else:
                smoothed_emotions[emotion] = values
        
        return {
            "polarity": smoothed_polarity.tolist(),
            "emotions": smoothed_emotions
        }

    def infer_genre_from_profile(self, profile, title=""):
        """Infer book genre from emotional profile (simplified)."""
        emotions = profile["overall_emotions"]
        
        # Get top emotions
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        top_emotions = [e for e, _ in top_emotions]
        
        # Very simplified genre inference
        if "fear" in top_emotions or "disgust" in top_emotions:
            return "Horror/Thriller"
        elif "love" in top_emotions or "desire" in top_emotions:
            return "Romance"
        elif "curiosity" in top_emotions or "realization" in top_emotions:
            return "Mystery"
        elif "joy" in top_emotions or "amusement" in top_emotions:
            return "Comedy"
        elif "sadness" in top_emotions or "grief" in top_emotions:
            return "Drama"
        else:
            return "Fiction"

    def extract_book_profile(self, title, text):
        """Extract a comprehensive book profile with all features."""
        logger.info(f"Extracting book profile for: {title}")
        
        # Clean Gutenberg metadata
        text = self.clean_gutenberg_text(text)
        
        # Segment book into chapters
        chapters = self.segment_by_chapter(text)
        logger.info(f"Book segmented into {len(chapters)} chapters")
        
        # Process full text
        all_tokens, all_sentences = self.preprocess_text(text)
        
        # Extract overall emotion profile
        emotion_profile = self.extract_lexicon_based_emotions(all_tokens)
        
        # Extract sentiment trajectory
        trajectory = self.create_sentiment_trajectory(chapters)
        
        # Apply smoothing
        smoothed_trajectory = self.apply_gaussian_smoothing(trajectory, sigma=1)
        
        # Extract TF-IDF features and topics
        tfidf_matrix, feature_names = self.extract_tfidf_features(chapters)
        topics, doc_topic_dist = self.extract_topics_with_lda(tfidf_matrix, feature_names)
        
        # Convert NumPy array to list
        if isinstance(doc_topic_dist, np.ndarray):
            doc_topic_dist = doc_topic_dist.tolist()
        
        # Extract character emotions
        logger.info("Extracting character emotions...")
        character_emotions = self.extract_character_emotions(text)
        
        # Infer genre
        genre = self.infer_genre_from_profile({
            "overall_emotions": emotion_profile["scores"]
        }, title=title)
        
        # Compile book profile
        book_profile = {
            "title": title,
            "genre": genre,
            "overall_emotions": emotion_profile["scores"],
            "sentiment_trajectory": {
                "raw": trajectory,
                "smoothed": smoothed_trajectory
            },
            "topics": topics,
            "chapter_topic_distribution": doc_topic_dist,
            "character_emotions": character_emotions
        }
        
        logger.info("Book profile extraction complete")
        return book_profile 