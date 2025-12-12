"""
Text Processing Utilities
Handles text chunking and processing for the chatbot.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import Config


class TextProcessor:
    """Class for processing and chunking text."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize TextProcessor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespaces
        text = " ".join(text.split())
        
        # Remove special characters that might interfere
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text.strip()
    
    def get_text_stats(self, text: str) -> dict:
        """
        Get statistics about the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        chunks = self.chunk_text(text)
        
        return {
            "total_characters": len(text),
            "total_words": len(text.split()),
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
        }
