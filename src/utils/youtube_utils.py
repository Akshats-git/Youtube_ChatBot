"""
YouTube Utilities
Handles YouTube video metadata retrieval and transcript extraction.
"""

from typing import Optional, Dict, List, Any
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube


def format_timestamp(seconds: float) -> str:
    """Format a number of seconds as MM:SS or HH:MM:SS."""
    total = int(max(seconds, 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class YouTubeUtils:
    """Utility class for YouTube video operations."""

    @staticmethod
    def extract_video_id(url_or_id: str) -> Optional[str]:
        """
        Extract a YouTube video ID from a URL or return it as-is if already an ID.

        Args:
            url_or_id: YouTube URL or bare video ID.

        Returns:
            11-character video ID, or None if it cannot be determined.
        """
        if not url_or_id:
            return None

        candidate = url_or_id.strip()
        if YouTubeUtils.validate_video_id(candidate):
            return candidate

        try:
            parsed = urlparse(candidate)
            hostname = (parsed.hostname or "").lower()

            if "youtu.be" in hostname:
                video_id = parsed.path.lstrip("/").split("?")[0]
                return video_id if YouTubeUtils.validate_video_id(video_id) else None

            if "youtube.com" in hostname:
                query_params = parse_qs(parsed.query)
                if "v" in query_params:
                    video_id = query_params["v"][0]
                    return video_id if YouTubeUtils.validate_video_id(video_id) else None

                path_parts = [p for p in parsed.path.split("/") if p]
                for marker in ("embed", "v", "shorts"):
                    if marker in path_parts:
                        idx = path_parts.index(marker)
                        if idx + 1 < len(path_parts):
                            video_id = path_parts[idx + 1]
                            return video_id if YouTubeUtils.validate_video_id(video_id) else None
        except Exception:
            return None

        return None

    @staticmethod
    def validate_video_id(video_id: str) -> bool:
        """Return True if the string is a valid 11-character YouTube video ID."""
        if not video_id:
            return False
        return len(video_id) == 11 and all(c.isalnum() or c in "-_" for c in video_id)

    @staticmethod
    def get_video_info(video_id: str) -> Dict[str, str]:
        """
        Fetch video metadata via pytube.

        Returns a dict with at minimum ``video_id``.  On failure an ``error``
        key is added so callers can detect degraded metadata without crashing.
        """
        if not YouTubeUtils.validate_video_id(video_id):
            return {
                "title": "YouTube Video",
                "video_id": video_id,
                "error": "Invalid video ID format",
            }

        try:
            video = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            return {
                "title": video.title or f"YouTube Video – {video_id}",
                "author": video.author or "Unknown",
                "length": str(video.length or 0),
                "views": str(video.views or 0),
                "description": video.description or "",
                "video_id": video_id,
            }
        except Exception as exc:
            return {
                "title": f"YouTube Video – {video_id}",
                "author": "Unknown",
                "length": "0",
                "views": "0",
                "description": "",
                "video_id": video_id,
                "error": str(exc),
            }

    @staticmethod
    def get_transcript_segments(
        video_id: str,
        languages: Optional[List[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch timestamped transcript segments for a video.

        Args:
            video_id: YouTube video ID.
            languages: Ordered list of BCP-47 language codes to try.

        Returns:
            List of ``{text, start, duration}`` dicts, or None on failure.
        """
        if languages is None:
            languages = ["en"]

        try:
            api = YouTubeTranscriptApi()
            transcript_obj = api.fetch(video_id=video_id, languages=languages)

            segments = []
            for snippet in transcript_obj:
                text = " ".join(snippet.text.split())
                if not text:
                    continue
                segments.append(
                    {
                        "text": text,
                        "start": float(snippet.start),
                        "duration": float(snippet.duration),
                    }
                )
            return segments
        except Exception as exc:
            print(f"Error fetching transcript for {video_id}: {exc}")
            return None

    @staticmethod
    def get_transcript(
        video_id: str,
        languages: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Return the full transcript as a single string, or None on failure."""
        segments = YouTubeUtils.get_transcript_segments(video_id, languages=languages)
        if not segments:
            return None
        return " ".join(segment["text"] for segment in segments)

    @staticmethod
    def get_available_transcripts(video_id: str) -> list:
        """Return the list of available transcript objects for a video."""
        try:
            api = YouTubeTranscriptApi()
            return list(api.list(video_id))
        except Exception as exc:
            print(f"Error listing transcripts for {video_id}: {exc}")
            return []
