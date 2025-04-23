# BookAlytics

An AI-powered book sentiment analysis web application that extracts emotional and thematic patterns from literary works.

## Features

- **Sentiment Analysis**: Track emotional polarity throughout a book's narrative
- **Emotion Detection**: Identify and visualize specific emotions across chapters
- **Character Analysis**: Extract emotional profiles for main characters
- **Topic Modeling**: Discover key themes and their distribution
- **Genre Inference**: Automatic detection of likely genre based on emotional patterns
- **Interactive Visualizations**: Beautiful, downloadable charts and graphs

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
4. Explore the visualizations and insights
5. Download individual charts or all visualizations as a ZIP file

## API Endpoints

BookAlytics also provides a REST API for programmatic access:

- `POST /api/analyze` - Upload and analyze a book
- `GET /api/jobs/{job_id}` - Check analysis job status
- `GET /api/results/{job_id}` - Get analysis results
- `GET /api/visualizations/{job_id}/{vis_type}` - Get specific visualization
- `POST /api/visualizations-data/{job_id}` - Get all visualizations as base64 encoded data

Full API documentation is available at http://localhost:8000/docs when the server is running.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project was inspired by computational literary analysis techniques
- Uses HuggingFace's transformers library for emotion detection
- Incorporates visualization techniques from various digital humanities projects