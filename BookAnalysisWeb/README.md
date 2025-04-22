# Book Sentiment Analysis Web Service

A web-based tool for analyzing emotional patterns, character sentiments, and thematic elements in books.

## Features

- **Upload books** in plain text (.txt) format
- **Analyze sentiment patterns** throughout the book
- **Track emotional trajectories** across chapters
- **Discover main characters** and their associated emotions
- **Identify key topics** and their distribution in the text
- **Visualize results** with interactive charts and graphs

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/BookAnalysisWeb.git
   cd BookAnalysisWeb
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Usage

1. Start the web server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload a book in .txt format and click "Analyze Book"

4. Wait for the analysis to complete and explore the results

## Project Structure

- `app.py`: Main Flask application
- `book_analyzer/`: Python package for text analysis
  - `analyzer.py`: Core book analysis functionality
- `static/`: Static files (CSS, JavaScript)
- `templates/`: HTML templates
- `uploads/`: Temporary storage for uploaded books
- `results/`: Storage for analysis results

## Development

To add new features or modify the analysis:

1. The main analysis logic is in `book_analyzer/analyzer.py`
2. Frontend functionality is in `static/js/main.js`
3. The interface is defined in `templates/index.html`

## Requirements

- Python 3.7+
- Flask
- NLTK
- scikit-learn
- matplotlib
- TextBlob

## License

This project is licensed under the MIT License - see the LICENSE file for details. 