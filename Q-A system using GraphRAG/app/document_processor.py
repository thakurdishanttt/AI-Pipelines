"""
Document processing module for the Knowledge Graph RAG system.
This module handles loading PDF documents and processing them into semantic chunks.
"""

from typing import List
import PyPDF2
from .utils import logger
from .semantic_chunking import semantic_chunk

class DocumentProcessor:
    """
    Class for processing PDF documents into semantic chunks.
    """

    def load_pdf(self, pdf_path: str) -> List[str]:
        """
        Load a PDF document and extract text from its pages.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[str]: List of text content from each page.
        """
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            return pages
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
            return []

    def process_pages(self, pages: List[str], sentence_model) -> List[str]:
        """
        Process the text content of PDF pages into semantic chunks.

        Args:
            pages (List[str]): List of text content from PDF pages.
            sentence_model: Pre-trained sentence embedding model.

        Returns:
            List[str]: List of semantic chunks.
        """
        all_chunks = []
        for page in pages:
            chunks = semantic_chunk(page, sentence_model)
            all_chunks.extend(chunks)
        return all_chunks
