import os
import tempfile
import uuid
import asyncio
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Depends, Path
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from ..core.extractor import BookFeatureExtractor
from ..core.model_cache import model_cache
from ..core.recommender import BookRecommender
from ..utils.file_handling import (
    load_book_from_bytes, 
    save_profile, 
    load_profile, 
    get_supported_file_extensions
)
from ..utils.visualizations import (
    plot_sentiment_trajectory, 
    plot_character_emotions, 
    plot_topic_heatmap,
    generate_all_visualizations
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create data directories
UPLOAD_DIR = "data/uploads"
RESULTS_DIR = "data/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="BookAlytics API",
    description="API for book sentiment and emotion analysis",
    version="1.0.0"
)

# Initialize model cache at startup
logger.info("Initializing model cache")
# Pre-load commonly used models
model_cache.get_spacy_model("en_core_web_sm")
emotion_model = "joeddav/distilbert-base-uncased-go-emotions-student"
model_cache.get_tokenizer(emotion_model)
model_cache.get_transformer_model(emotion_model)
logger.info("Model cache initialized")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create analysis job storage
JOBS = {}

# Create a storage for book profiles to enable recommendations
BOOK_PROFILES = []
recommender = None

class JobStatus(BaseModel):
    """Model for job status response."""
    job_id: str
    status: str
    progress: Optional[float] = 0.0
    message: Optional[str] = None
    result_url: Optional[str] = None
    created_at: datetime

class AnalysisResponse(BaseModel):
    """Model for analysis response."""
    job_id: str
    status: str
    message: str

class VisualizationRequest(BaseModel):
    """Model for visualization request."""
    job_id: str
    visualization_type: str  # 'sentiment', 'characters', 'topics', or 'all'

class JobResult(BaseModel):
    """Model for job result."""
    title: str
    genre: str
    overall_emotions: Dict[str, float]
    character_emotions: Dict[str, Dict[str, float]]
    topics: Dict[str, List[str]]
    visualizations: Optional[Dict[str, str]] = None  # Base64 encoded images

@app.get("/", response_class=HTMLResponse)
async def root():
    """API root endpoint with basic information."""
    return """
    <html>
        <head>
            <title>BookAlytics API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #2c3e50;
                }
                code {
                    background-color: #f8f9fa;
                    padding: 2px 4px;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <h1>BookAlytics API</h1>
            <p>Welcome to the BookAlytics API for book sentiment and emotion analysis.</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><code>POST /analyze</code> - Upload and analyze a book</li>
                <li><code>GET /jobs/{job_id}</code> - Check job status</li>
                <li><code>GET /results/{job_id}</code> - Get analysis results</li>
                <li><code>GET /visualizations/{job_id}</code> - Get visualizations</li>
            </ul>
            <p>For API documentation, visit <a href="/docs">/docs</a>.</p>
        </body>
    </html>
    """

@app.get("/supported-formats")
async def supported_formats():
    """Get supported file formats."""
    extensions = get_supported_file_extensions()
    return {"supported_formats": extensions}

async def process_book(job_id: str, file_content: bytes, filename: str):
    """
    Process a book file in the background.
    
    Args:
        job_id (str): Job ID
        file_content (bytes): File content
        filename (str): Filename
    """
    try:
        # Update job status
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["progress"] = 10.0
        JOBS[job_id]["message"] = "Loading book file..."
        
        # Load the book
        title, text = load_book_from_bytes(file_content, filename)
        JOBS[job_id]["progress"] = 20.0
        JOBS[job_id]["message"] = "Book loaded successfully. Extracting features..."
        
        # Create extractor
        extractor = BookFeatureExtractor()
        
        # Extract features
        book_profile = extractor.extract_book_profile(title, text)
        JOBS[job_id]["progress"] = 70.0
        JOBS[job_id]["message"] = "Features extracted. Generating visualizations..."
        
        # Create results directory
        results_path = os.path.join(RESULTS_DIR, job_id)
        os.makedirs(results_path, exist_ok=True)
        
        # Save profile
        try:
            profile_path = os.path.join(results_path, "profile.json")
            save_profile(book_profile, profile_path)
            logger.info(f"Profile saved successfully to {profile_path}")
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
            raise
        
        # Generate visualizations
        try:
            generate_all_visualizations(book_profile, output_dir=results_path)
            logger.info("Visualizations generated successfully")
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            # Continue even if visualization fails
        
        JOBS[job_id]["progress"] = 90.0
        JOBS[job_id]["message"] = "Visualizations generated. Finalizing results..."
        
        # Update job status
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100.0
        JOBS[job_id]["message"] = "Analysis completed successfully"
        JOBS[job_id]["result_url"] = f"/results/{job_id}"
        
    except Exception as e:
        logger.error(f"Error processing book: {str(e)}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = f"Error processing book: {str(e)}"

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and analyze a book file.
    
    Args:
        file (UploadFile): Book file to analyze
        
    Returns:
        AnalysisResponse: Job information
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Check file extension
        filename = file.filename
        _, extension = os.path.splitext(filename)
        
        if extension.lower() not in get_supported_file_extensions():
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {get_supported_file_extensions()}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Store job information
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued. Waiting for processing...",
            "created_at": datetime.now(),
            "filename": filename
        }
        
        # Start processing in background
        background_tasks.add_task(process_book, job_id, file_content, filename)
        
        return AnalysisResponse(
            job_id=job_id,
            status="queued",
            message="Book uploaded successfully. Processing started."
        )
        
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get the status of a job.
    
    Args:
        job_id (str): Job ID
        
    Returns:
        JobStatus: Job status information
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**JOBS[job_id])

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """
    Get the results of a completed job.
    
    Args:
        job_id (str): Job ID
        
    Returns:
        dict: Analysis results
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job['status']}"
        )
    
    # Load profile
    results_path = os.path.join(RESULTS_DIR, job_id)
    profile_path = os.path.join(results_path, "profile.json")
    
    try:
        profile = load_profile(profile_path)
        
        # Create response object
        result = JobResult(
            title=profile["title"],
            genre=profile["genre"],
            overall_emotions=profile["overall_emotions"],
            character_emotions=profile["character_emotions"],
            topics=profile["topics"]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading results: {str(e)}")

@app.get("/visualizations/{job_id}/{vis_type}")
async def get_visualization(
    job_id: str, 
    vis_type: str = Path(..., description="Visualization type: sentiment, characters, topics")
):
    """
    Get a specific visualization for a job.
    
    Args:
        job_id (str): Job ID
        vis_type (str): Visualization type
        
    Returns:
        FileResponse: Image file
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job['status']}"
        )
    
    # Map visualization type to filename
    filename_map = {
        "sentiment": "polarity_trajectory.png",
        "emotions": "emotion_trajectory.png",
        "composition": "emotion_composition.png",
        "characters": "character_emotions.png",
        "topics": "topic_heatmap.png"
    }
    
    if vis_type not in filename_map:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid visualization type. Supported types: {list(filename_map.keys())}"
        )
    
    # Get file path
    results_path = os.path.join(RESULTS_DIR, job_id)
    file_path = os.path.join(results_path, filename_map[vis_type])
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(file_path)

@app.post("/visualizations-data/{job_id}")
async def get_visualizations_data(job_id: str):
    """
    Get all visualizations as base64 encoded data.
    
    Args:
        job_id (str): Job ID
        
    Returns:
        dict: Visualization data
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job['status']}"
        )
    
    # Load profile
    results_path = os.path.join(RESULTS_DIR, job_id)
    profile_path = os.path.join(results_path, "profile.json")
    
    try:
        profile = load_profile(profile_path)
        
        # Generate visualizations
        visualizations = generate_all_visualizations(profile, return_base64=True)
        
        return {"visualizations": visualizations}
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")

@app.post("/admin/clear-cache")
async def clear_model_cache():
    """
    Clear the model cache to free up memory.
    Only use this endpoint if the server is running low on memory.
    
    Returns:
        dict: Status message
    """
    try:
        logger.info("Clearing model cache")
        model_cache.clear_cache()
        
        # Manually trigger garbage collection
        import gc
        gc.collect()
        
        # Re-initialize commonly used models
        model_cache.get_spacy_model("en_core_web_sm")
        
        return {"status": "success", "message": "Model cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing model cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing model cache: {str(e)}")

@app.post("/results/{job_id}")
async def save_results(job_id: str):
    """
    Save the results of a completed job to the book profiles database
    for future recommendations.
    
    Args:
        job_id (str): Job ID
        
    Returns:
        dict: Status message
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job['status']}"
        )
    
    # Load profile
    results_path = os.path.join(RESULTS_DIR, job_id)
    profile_path = os.path.join(results_path, "profile.json")
    
    try:
        profile = load_profile(profile_path)
        
        # Check if this book is already in the profiles
        existing_idx = -1
        for i, existing in enumerate(BOOK_PROFILES):
            if existing["title"] == profile["title"]:
                existing_idx = i
                break
        
        if existing_idx >= 0:
            # Update existing profile
            BOOK_PROFILES[existing_idx] = profile
            logger.info(f"Updated existing profile for {profile['title']}")
        else:
            # Add new profile
            BOOK_PROFILES.append(profile)
            logger.info(f"Added new profile for {profile['title']}")
        
        # Create/update recommender
        global recommender
        recommender = BookRecommender(BOOK_PROFILES)
        
        return {"message": "Profile saved to recommendations database"}
        
    except Exception as e:
        logger.error(f"Error saving to recommendations database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving to recommendations database: {str(e)}")

@app.get("/recommendations/{book_title}")
async def get_recommendations(book_title: str):
    """
    Get book recommendations based on a book title.
    
    Args:
        book_title (str): Book title to get recommendations for
        
    Returns:
        dict: Recommendations
    """
    global recommender
    
    try:
        if not BOOK_PROFILES:
            return JSONResponse(
                status_code=400,
                content={"detail": "No books in the recommendation database. Please analyze books first."}
            )
        
        if recommender is None:
            # Initialize the recommender if not already done
            recommender = BookRecommender(BOOK_PROFILES)
        
        recommendations = recommender.get_recommendations_for_book(book_title)
        
        if "error" in recommendations:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Book '{book_title}' not found in the recommendation database."}
            )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting recommendations: {str(e)}"}
        )

@app.get("/books")
async def get_books():
    """
    Get all books in the recommendation database.
    
    Returns:
        list: List of book titles and genres
    """
    books = [{"title": book["title"], "genre": book.get("genre", "unknown")} 
            for book in BOOK_PROFILES]
    
    return {"books": books}

def start_server():
    """Start the FastAPI server."""
    uvicorn.run("bookalytics.api.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start_server() 