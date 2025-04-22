from flask import Flask, render_template, request, jsonify, send_from_directory, session
import os
import uuid
from werkzeug.utils import secure_filename
import threading
import json
from book_analyzer.analyzer import BookAnalyzer

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['SECRET_KEY'] = 'your-secret-key-here'  # For session management
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Store analysis progress
analysis_progress = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{analysis_id}_{filename}")
        file.save(file_path)
        
        # Create results directory for this analysis
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Initialize progress
        analysis_progress[analysis_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting analysis...'
        }
        
        # Start analysis in a background thread
        thread = threading.Thread(
            target=analyze_book_task,
            args=(file_path, result_dir, analysis_id, filename)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'analysis_id': analysis_id,
            'filename': filename
        })

def analyze_book_task(file_path, result_dir, analysis_id, filename):
    try:
        # Update progress
        analysis_progress[analysis_id]['message'] = 'Loading book...'
        analysis_progress[analysis_id]['progress'] = 10
        
        # Initialize analyzer
        analyzer = BookAnalyzer()
        
        # Load and preprocess book
        analysis_progress[analysis_id]['message'] = 'Processing text...'
        analysis_progress[analysis_id]['progress'] = 20
        
        # Extract book title from filename
        title = os.path.splitext(filename)[0]
        
        # Load book
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            book_text = f.read()
        
        # Extract features
        analysis_progress[analysis_id]['message'] = 'Extracting sentiment...'
        analysis_progress[analysis_id]['progress'] = 40
        
        book_profile = analyzer.analyze_book(title, book_text, 
                                             progress_callback=lambda msg, pct: update_progress(analysis_id, msg, pct))
        
        # Save results
        with open(os.path.join(result_dir, 'profile.json'), 'w', encoding='utf-8') as f:
            json.dump(book_profile, f, indent=2)
        
        # Generate visualizations
        analysis_progress[analysis_id]['message'] = 'Generating visualizations...'
        analysis_progress[analysis_id]['progress'] = 80
        
        analyzer.generate_visualizations(book_profile, result_dir)
        
        # Mark as complete
        analysis_progress[analysis_id]['status'] = 'completed'
        analysis_progress[analysis_id]['progress'] = 100
        analysis_progress[analysis_id]['message'] = 'Analysis complete!'
        
    except Exception as e:
        analysis_progress[analysis_id]['status'] = 'failed'
        analysis_progress[analysis_id]['message'] = f'Error: {str(e)}'
        print(f"Analysis error: {str(e)}")

def update_progress(analysis_id, message, progress):
    if analysis_id in analysis_progress:
        analysis_progress[analysis_id]['message'] = message
        analysis_progress[analysis_id]['progress'] = progress

@app.route('/progress/<analysis_id>', methods=['GET'])
def get_progress(analysis_id):
    if analysis_id in analysis_progress:
        return jsonify(analysis_progress[analysis_id])
    return jsonify({'error': 'Analysis not found'}), 404

@app.route('/results/<analysis_id>', methods=['GET'])
def get_results(analysis_id):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)
    
    if not os.path.exists(result_dir):
        return jsonify({'error': 'Results not found'}), 404
    
    # Read profile data
    try:
        with open(os.path.join(result_dir, 'profile.json'), 'r', encoding='utf-8') as f:
            profile = json.load(f)
    except:
        return jsonify({'error': 'Results file corrupted or not found'}), 500
    
    # Get list of visualization files
    visualizations = []
    for file in os.listdir(result_dir):
        if file.endswith(('.png', '.jpg', '.svg')):
            visualizations.append(file)
    
    return jsonify({
        'status': 'success',
        'profile': profile,
        'visualizations': visualizations
    })

@app.route('/results/<analysis_id>/visualization/<filename>', methods=['GET'])
def get_visualization(analysis_id, filename):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)
    return send_from_directory(result_dir, filename)

if __name__ == '__main__':
    app.run(debug=True) 