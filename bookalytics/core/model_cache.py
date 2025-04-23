import os
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Singleton class for caching NLP models to avoid reloading them for each request.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating new ModelCache instance")
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.tokenizers = {}
            cls._instance.nlp_models = {}
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create cache directory
            cache_dir = Path("data/model_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cls._instance.cache_dir = cache_dir
            
            logger.info(f"Model cache initialized with device: {cls._instance.device}")
        return cls._instance
    
    def get_transformer_model(self, model_name):
        """Get cached transformer model or load if not available."""
        if model_name in self.models:
            logger.info(f"Using cached model: {model_name}")
            return self.models[model_name]
        
        logger.info(f"Loading model: {model_name}")
        from transformers import AutoModelForSequenceClassification
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir / model_name.replace('/', '_'))
            ).to(self.device)
            self.models[model_name] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def get_tokenizer(self, model_name):
        """Get cached tokenizer or load if not available."""
        if model_name in self.tokenizers:
            logger.info(f"Using cached tokenizer: {model_name}")
            return self.tokenizers[model_name]
        
        logger.info(f"Loading tokenizer: {model_name}")
        from transformers import AutoTokenizer
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir / model_name.replace('/', '_'))
            )
            self.tokenizers[model_name] = tokenizer
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer {model_name}: {str(e)}")
            return None
    
    def get_spacy_model(self, model_name):
        """Get cached spaCy model or load if not available."""
        if model_name in self.nlp_models:
            logger.info(f"Using cached spaCy model: {model_name}")
            return self.nlp_models[model_name]
        
        logger.info(f"Loading spaCy model: {model_name}")
        import spacy
        
        try:
            nlp = spacy.load(model_name)
            self.nlp_models[model_name] = nlp
            return nlp
        except OSError:
            logger.warning(f"Downloading spaCy model: {model_name}")
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            self.nlp_models[model_name] = nlp
            return nlp
        except Exception as e:
            logger.error(f"Error loading spaCy model {model_name}: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear all cached models from memory."""
        logger.info("Clearing model cache")
        self.models = {}
        self.tokenizers = {}
        self.nlp_models = {}
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Initialize global instance
model_cache = ModelCache() 