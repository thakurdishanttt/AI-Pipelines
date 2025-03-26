"""
Configuration file for the Knowledge Graph RAG system.
This file manages constants and environment variables required for the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
CHROMA_PATH = "./chroma_vectordb_openAI1207"  # Path to store Chroma vector database
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 1  # Delay (in seconds) between retries
MAX_CONTEXT_LENGTH = 15000  # Maximum context length for processing
CHUNK_SIZE = 1000  # Chunk size for document processing
SIMILARITY_THRESHOLD = 0.7  # Threshold for semantic similarity
SEMANTIC_WEIGHT = 0.8  # Weight for semantic similarity edges in the graph
SEQUENTIAL_WEIGHT = 0.6  # Weight for sequential edges in the graph

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # API key loaded from .env file
