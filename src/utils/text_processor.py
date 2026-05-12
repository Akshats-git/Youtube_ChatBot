"""
Text Processing Utilities
Handles text chunking and processing for the chatbot.
"""

from bisect import bisect_right
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
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

    def build_documents_from_segments(
        self,
        segments: List[Dict[str, Any]],
        video_id: str,
        language: str,
        video_title: str = ""
    ) -> List[Document]:
        """
        Convert transcript segments into chunked LangChain Documents with timestamps.

        Args:
            segments: Transcript segments with text, start, duration
            video_id: YouTube video ID
            language: Transcript language code
            video_title: Optional video title

        Returns:
            List of Document chunks with metadata
        """
        if not segments:
            return []

        full_text, segment_offsets, segment_times = self._flatten_segments(segments)
        if not segment_offsets:
            return []
        chunks = self.text_splitter.split_text(full_text)

        documents: List[Document] = []
        search_cursor = 0

        for chunk_id, chunk in enumerate(chunks):
            if not chunk:
                continue

            start_index = full_text.find(chunk, search_cursor)
            if start_index == -1:
                start_index = search_cursor

            end_index = start_index + len(chunk)
            search_cursor = end_index

            start_seg_idx = max(bisect_right(segment_offsets, start_index) - 1, 0)
            end_seg_idx = max(bisect_right(segment_offsets, end_index) - 1, 0)

            start_time, _ = segment_times[start_seg_idx]
            _, end_time = segment_times[end_seg_idx]

            metadata = {
                "chunk_id": chunk_id,
                "video_id": video_id,
                "language": language,
                "video_title": video_title,
                "source": "youtube_transcript",
                "start_time": start_time,
                "end_time": end_time,
                "char_start": start_index,
                "char_end": end_index,
                "text_length": len(chunk)
            }

            documents.append(Document(page_content=chunk, metadata=metadata))

        return documents

    def _flatten_segments(
        self,
        segments: List[Dict[str, Any]]
    ) -> Tuple[str, List[int], List[Tuple[float, float]]]:
        """
        Flatten transcript segments into a single string while tracking offsets.

        Returns:
            Tuple of full text, segment start offsets, and (start, end) times
        """
        parts: List[str] = []
        offsets: List[int] = []
        times: List[Tuple[float, float]] = []
        cursor = 0

        for segment in segments:
            text = " ".join(segment.get("text", "").split())
            if not text:
                continue

            offsets.append(cursor)
            start_time = float(segment.get("start", 0.0))
            duration = float(segment.get("duration", 0.0))
            end_time = start_time + duration
            times.append((start_time, end_time))

            parts.append(text)
            cursor += len(text) + 1

        full_text = " ".join(parts)
        return full_text, offsets, times
    
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
