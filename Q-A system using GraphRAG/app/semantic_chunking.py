"""
Semantic chunking module for the Knowledge Graph RAG system.
This module provides functions to split text into semantic chunks using clustering techniques.
"""

from typing import List
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
from .utils import logger

def simple_sentence_split(text: str) -> List[str]:
    """
    Simple sentence splitting fallback method.

    Args:
        text (str): Input text to split.

    Returns:
        List[str]: List of sentences.
    """
    # Split on common sentence endings
    sentences = []
    current = ""

    # Add space after common sentence endings if missing
    text = text.replace('.',' . ').replace('!',' ! ').replace('?',' ? ')
    text = ' '.join(text.split())  # Normalize spaces

    for word in text.split():
        current += word + " "
        if word in ['.', '!', '?']:
            current = current.strip()
            if current:
                sentences.append(current)
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences or [text]

def semantic_chunk(text: str, sentence_model, max_chunk_size: int = 750, min_chunk_size: int = 200) -> List[str]:
    """
    Split text into semantic chunks using sentence embeddings and clustering.

    Args:
        text (str): Input text to chunk.
        sentence_model: Pre-trained sentence embedding model.
        max_chunk_size (int): Maximum size of a chunk.
        min_chunk_size (int): Minimum size of a chunk.

    Returns:
        List[str]: List of semantic chunks.
    """
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return [text] if text.strip() else []

        # Remove empty sentences and normalize whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        embeddings = sentence_model.encode(sentences)
        text_length = sum(len(s) for s in sentences)
        num_clusters = max(1, text_length // max_chunk_size)

        kmeans = KMeans(n_clusters=min(num_clusters, len(sentences)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        grouped_sentences = defaultdict(list)
        for sentence, cluster in zip(sentences, clusters):
            grouped_sentences[cluster].append(sentence)

        chunks = []
        for cluster_sentences in grouped_sentences.values():
            chunks.append(' '.join(cluster_sentences))
        return chunks

    except Exception as e:
        logger.error(f"Error in semantic chunking: {e}")
        return [text] if text.strip() else []