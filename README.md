# BookAlytics

An AI-powered book sentiment analysis web application that extracts emotional and thematic patterns from literary works.

## Features

- **Sentiment Analysis**: Track emotional polarity throughout a book's narrative
- **Emotion Detection**: Identify and visualize specific emotions across chapters
- **Character Analysis**: Extract emotional profiles for main characters
- **Topic Modeling**: Discover key themes and their distribution
- **Genre Inference**: Automatic detection of likely genre based on emotional patterns
- **Book Recommendations**: Get personalized recommendations based on emotional profiles and genres
- **Interactive Visualizations**: Beautiful, downloadable charts and graphs
- **Advanced Chapter Detection**: Smart recognition of chapters using various formats, including written-out numbers (e.g., "Chapter One" to "Chapter One Hundred")

## Architecture

BookAlytics uses a multi-model approach that combines various NLP techniques:

- **BERT-based Emotion Analysis**: Fine-tuned transformer models to detect complex emotions
- **Character Recognition**: SpaCy-powered NER to identify and track main characters
- **Topic Modeling**: LDA (Latent Dirichlet Allocation) for thematic analysis
- **Temporal Analysis**: Sentiment trajectory tracking with Gaussian smoothing
- **Recommendation System**: Clustering-based book recommendations using genre and emotional profiles

## Methodology and Technical Approach

This application was developed through a systematic approach, beginning with exploratory data analysis in Jupyter notebooks before transforming the insights into a production-ready application. The methodology involved several key components:

### 1. Multi-modal Emotion Analysis

BookAlytics employs a dual approach to emotion detection:

- **Lexicon-based Method**: Utilizes predefined emotion dictionaries for baseline detection, where words are associated with specific emotions. This provides a foundation for basic emotion recognition.
  
- **Transformer-based Deep Learning**: Leverages the `joeddav/distilbert-base-uncased-go-emotions-student` model, a distilled version of BERT fine-tuned specifically for emotion detection. This model can identify 28 distinct emotions with nuanced understanding of context.

The transformer approach was selected over traditional machine learning algorithms because:
1. It captures contextual information rather than just word presence
2. It understands emotional nuances across different literary styles
3. It performs well even with the complex language patterns found in literature

### 2. Character Analysis Innovation

Character analysis employs a novel pipeline:

1. **Entity Recognition**: Uses SpaCy's NER to identify character names in the text
2. **Name Coreference Resolution**: Employs fuzzy matching (via `difflib`) to connect variations of the same character name
3. **Contextual Analysis**: Extracts paragraphs where characters appear to create a "character context corpus"
4. **Character-Specific Emotion Extraction**: Applies the emotion model to each character's context corpus

This approach allows analysis of character arcs and relationships throughout the narrative.

### 3. Text Segmentation and Preprocessing

The text segmentation approach was carefully designed to handle various book formats:

- **Chapter Detection**: Uses intelligent regex patterns to identify chapter headings in multiple formats, including numbered chapters, Roman numerals, and written-out numbers ("Chapter One" to "Chapter One Hundred")
- **Fallback Mechanisms**: Implements progressive fallbacks for books with non-standard chapter markings
- **Gutenberg Cleanup**: Automatically removes Project Gutenberg metadata to focus on the actual literary content

Preprocessing includes:
- Stop word removal and lemmatization using NLTK
- Tokenization at both word and sentence levels
- Custom handling of literary punctuation and dialogue

### 4. Temporal Analysis with Signal Processing

To create emotion trajectories:

1. Emotions are extracted at the chapter level
2. Raw trajectories often contain noise, so Gaussian smoothing (from SciPy) is applied
3. Signal processing techniques help identify significant emotional shifts
4. The result is a continuous curve showing emotional development through the narrative

### 5. Topic Modeling and Theme Extraction

The LDA (Latent Dirichlet Allocation) approach was chosen after evaluating several options:

- **CountVectorizer with Bigrams**: Captures phrases and multi-word concepts
- **Custom Stopword Generation**: Dynamically identifies character names and removes them from topic modeling to prevent character names from dominating topics
- **LDA Optimization**: Parameters fine-tuned through experimentation in the notebook stage

### 6. Memory and Performance Optimization

To make these complex models practical in a web application:

- **Model Caching**: Implements a singleton pattern to load models once and reuse them across requests
- **Batch Processing**: Processes text in chunks to avoid memory issues with long books
- **Device Management**: Automatically utilizes GPU acceleration when available
- **Lazy Loading**: Loads specific models only when needed

### 7. Recommendation System Development

The recommendation engine uses:

- **Multi-feature Vectors**: Combines emotion profiles and genre information
- **Clustering Algorithms**: Groups similar books based on their emotional fingerprints
- **Hybrid Recommendation**: Balances between emotional similarity and genre matching

This methodology, developed through careful experimentation in Jupyter notebooks, enabled the creation of a robust application that can process and analyze books in various formats, providing meaningful insights into their emotional and thematic content.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Option 1: Install from source

1. Clone the repository:

```bash
git clone https://github.com/your-username/bookalytics.git
cd bookalytics
```

2. Install the package and dependencies:

```bash
pip install -e .
```

This will install all required dependencies and the `bookalytics` command-line tool.

### Option 2: Use requirements.txt

1. Clone the repository:

```bash
git clone https://github.com/your-username/bookalytics.git
cd bookalytics
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required NLTK data:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

4. Download SpaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Running the Application

### Using the command-line tool (if installed with `pip install -e .`):

```bash
bookalytics
```

### Running the module directly:

```bash
python -m bookalytics.main
```

The application will start, and you can access it at http://localhost:8000 in your web browser.

## Usage

1. Open the application in your browser at http://localhost:8000
2. Upload a book file (.txt or .pdf format)
3. Wait for the analysis to complete (this may take several minutes for larger books)
4. Explore the visualizations and insights:
   - Summary shows key emotions without numerical scores
   - Character emotions display reveals how characters feel throughout the narrative
   - Trajectory charts show emotional patterns throughout the book
   - Topic analysis reveals key themes and their distribution
   - Book recommendations suggest similar works based on genre and emotional profile
5. Download individual charts or all visualizations as a ZIP file

## API Endpoints

BookAlytics provides a REST API for programmatic access:

- `POST /api/analyze` - Upload and analyze a book
- `GET /api/jobs/{job_id}` - Check analysis job status
- `GET /api/results/{job_id}` - Get analysis results
- `GET /api/visualization/{job_id}/{vis_type}` - Get specific visualization
- `GET /api/visualizations-data/{job_id}` - Get all visualizations as base64 encoded data
- `GET /api/recommendations/{book_title}` - Get book recommendations based on title
- `GET /api/books` - Get list of all analyzed books
- `POST /api/clear-cache` - Clear model cache
- `POST /api/save/{job_id}` - Save analysis results permanently

Full API documentation is available at http://localhost:8000/docs when the server is running.

## Technical Details

### Model Caching

BookAlytics implements intelligent model caching to optimize memory usage and improve performance. Large transformer models and NLP pipelines are loaded once and reused across multiple analyses.

### Error Handling

The application includes robust error handling for various scenarios:
- Graceful fallbacks for chapter detection if standard patterns aren't found
- JSON serialization safeguards for NumPy objects in the recommendation system
- Client-side error handling with meaningful user feedback

### Frontend Design

The frontend features a responsive design with a notebook-like appearance:
- Tabbed interface for easy navigation between different analysis views
- Interactive charts with hover details
- Collapsible sections for detailed exploration
- Mobile-friendly layout with responsive components

## License

[MIT License](LICENSE)

## Acknowledgments

- This project was inspired by computational literary analysis techniques
- Uses HuggingFace's transformers library for emotion detection
- Incorporates visualization techniques from various digital humanities projects
- Special thanks to the NLTK, SpaCy, and scikit-learn communities