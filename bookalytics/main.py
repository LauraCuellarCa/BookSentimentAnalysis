import os
import logging
from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .api.app import app as api_app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create main FastAPI app
app = FastAPI(
    title="BookAlytics",
    description="Book sentiment and emotion analysis web application",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.mount("/api", api_app)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the frontend static files
static_dir = os.path.join(script_dir, "frontend", "static")

# Verify the static directory exists
if os.path.isdir(static_dir):
    logger.info(f"Mounting static files from: {static_dir}")
    # Mount static files
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")
    # Create a simple redirect to the API if frontend is not available
    @app.get("/")
    async def redirect_to_api():
        return RedirectResponse(url="/api")

# Root redirect to frontend
@app.get("/")
async def root():
    return RedirectResponse(url="/index.html")

def start():
    """Start the application using uvicorn."""
    import uvicorn
    
    # Create data directories if they don't exist
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    
    logger.info("Starting BookAlytics server...")
    uvicorn.run("bookalytics.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start() 