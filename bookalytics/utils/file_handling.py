import os
import json
import pdfplumber
from PyPDF2 import PdfReader
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_book(file_path, encoding='utf-8'):
    """
    Load a book from a text file.
    
    Args:
        file_path (str): Path to the book file
        encoding (str): File encoding
        
    Returns:
        str: Book text
    """
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading book file: {str(e)}")
        raise

def load_book_from_pdf(file_path):
    """
    Extract text from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error loading PDF file with pdfplumber: {str(e)}")
        
        # Fall back to PyPDF2
        try:
            logger.info("Trying alternative PDF extraction with PyPDF2")
            text = ""
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e2:
            logger.error(f"Error loading PDF file with PyPDF2: {str(e2)}")
            raise

def load_book_from_bytes(file_bytes, filename, encodings=None):
    """
    Load a book from uploaded bytes data.
    
    Args:
        file_bytes (bytes): File content as bytes
        filename (str): Original filename
        encodings (list): List of encodings to try
        
    Returns:
        tuple: (title, text)
    """
    if encodings is None:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    # Extract title from filename
    title, extension = os.path.splitext(filename)
    
    # Handle PDF files
    if extension.lower() == '.pdf':
        # Save temporary file
        temp_path = f"temp_{filename}"
        with open(temp_path, 'wb') as f:
            f.write(file_bytes)
        
        # Extract text from PDF
        try:
            text = load_book_from_pdf(temp_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return title, text
        except Exception as e:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    # Handle text files
    else:
        # Try different encodings
        for encoding in encodings:
            try:
                text = file_bytes.decode(encoding)
                logger.info(f"Successfully decoded with {encoding} encoding")
                return title, text
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode with {encoding}")
        
        # If all encodings fail
        raise ValueError("Could not decode the book file with any of the attempted encodings")

def load_profile(profile_path):
    """
    Load a book profile from JSON.
    
    Args:
        profile_path (str): Path to the profile file
        
    Returns:
        dict: Book profile
    """
    with open(profile_path, 'r') as f:
        return json.load(f)

def convert_numpy_to_python(obj):
    """
    Convert NumPy arrays and other non-JSON serializable objects to Python types.
    
    Args:
        obj: Any Python object that might contain NumPy arrays
        
    Returns:
        obj: The object with NumPy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj

def save_profile(profile, output_path):
    """
    Save the extracted profile as JSON.
    
    Args:
        profile (dict): Book profile data
        output_path (str): Path to save the profile
        
    Returns:
        str: Output path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert NumPy arrays to Python lists
    profile = convert_numpy_to_python(profile)
    
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    logger.info(f"Book profile saved to {output_path}")
    return output_path

def get_supported_file_extensions():
    """
    Get list of supported file extensions.
    
    Returns:
        list: Supported file extensions
    """
    return ['.txt', '.pdf'] 