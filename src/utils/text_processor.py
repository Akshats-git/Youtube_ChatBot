"""
Text Processing Utilities
Chunking and document construction for transcript-based RAG.
"""

from bisect import bisect_right
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import Config


class TextProcessor:
    """Splits transcript text into overlapping chunks and builds LangChain Documents."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str) -> List[str]:
        """Split *text* into overlapping chunks."""
        if not text:
            return []
        return self.text_splitter.split_text(text)

    def preprocess_text(self, text: str) -> str:
        """Normalise whitespace and strip control characters."""
        return " ".join(text.split())

    def build_documents_from_segments(
        self,
        segments: List[Dict[str, Any]],
        video_id: str,
        language: str,
        video_title: str = "",
    ) -> List[Document]:
        """
        Convert timestamped transcript segments into chunked LangChain Documents.

        Each Document carries ``start_time`` / ``end_time`` metadata derived from
        the segment(s) that the chunk overlaps.

        Args:
            segments: List of ``{text, start, duration}`` dicts.
            video_id: YouTube video ID.
            language: BCP-47 language code.
            video_title: Optional video title for metadata.

        Returns:
            List of Documents ready for embedding.
        """
        if not segments:
            return []

        full_text, segment_offsets, segment_times = self._flatten_segments(segments)
        if not segment_offsets:
            return []

        documents: List[Document] = []
        search_cursor = 0

        for chunk_id, chunk in enumerate(self.text_splitter.split_text(full_text)):
            if not chunk:
                continue

            start_index = full_text.find(chunk, search_cursor)
            if start_index == -1:
                start_index = search_cursor
            end_index = start_index + len(chunk)
            search_cursor = end_index

            start_seg = max(bisect_right(segment_offsets, start_index) - 1, 0)
            end_seg = max(bisect_right(segment_offsets, end_index) - 1, 0)
            start_time, _ = segment_times[start_seg]
            _, end_time = segment_times[end_seg]

            documents.append(Document(
                page_content=chunk,
                metadata={
                    "chunk_id": chunk_id,
                    "video_id": video_id,
                    "language": language,
                    "video_title": video_title,
                    "source": "youtube_transcript",
                    "start_time": start_time,
                    "end_time": end_time,
                    "char_start": start_index,
                    "char_end": end_index,
                    "text_length": len(chunk),
                },
            ))

        return documents

    def _flatten_segments(
        self,
        segments: List[Dict[str, Any]],
    ) -> Tuple[str, List[int], List[Tuple[float, float]]]:
        """
        Join segments into a single string and record per-segment byte offsets.

        Returns:
            (full_text, character_offsets, (start_time, end_time) pairs)
        """
        parts: List[str] = []
        offsets: List[int] = []
        times: List[Tuple[float, float]] = []
        cursor = 0

        for seg in segments:
            text = " ".join(seg.get("text", "").split())
            if not text:
                continue
            start = float(seg.get("start", 0.0))
            duration = float(seg.get("duration", 0.0))
            offsets.append(cursor)
            times.append((start, start + duration))
            parts.append(text)
            cursor += len(text) + 1  # +1 for the space added by " ".join

        return " ".join(parts), offsets, times
