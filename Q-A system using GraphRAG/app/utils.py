"""
Utility functions for the Knowledge Graph RAG system.
Includes logging configuration and NLTK setup for natural language processing.
"""

import logging
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audit_analysis.log'),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ],
)
logger = logging.getLogger(__name__)

def setup_nltk():
    """
    Set up NLTK by downloading required resources.

    Returns:
        bool: True if setup is successful, False otherwise.
    """
    try:
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        logger.warning(f"NLTK setup failed: {e}")
        return False
