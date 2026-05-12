"""
YouTube Utilities Module
Handles YouTube video information extraction and transcript retrieval.
"""

from typing import Optional, Dict, List, Any
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube


class YouTubeUtils:
    """Utility class for YouTube video operations."""

    @staticmethod
    def extract_video_id(url_or_id: str) -> Optional[str]:
        """
        Extract a YouTube video ID from a URL or return the ID if already provided.

        Args:
            url_or_id: YouTube URL or video ID

        Returns:
            11-character video ID or None if not found
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
                video_id = parsed.path.lstrip("/")
                return video_id if YouTubeUtils.validate_video_id(video_id) else None

            if "youtube.com" in hostname:
                query_params = parse_qs(parsed.query)
                if "v" in query_params:
                    video_id = query_params["v"][0]
                    return video_id if YouTubeUtils.validate_video_id(video_id) else None

                path_parts = [part for part in parsed.path.split("/") if part]
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
        """
        Validate if the provided string is a valid YouTube video ID.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            True if valid, False otherwise
        """
        # YouTube video IDs are 11 characters long and contain alphanumeric, hyphen, and underscore
        if not video_id:
            return False
        return len(video_id) == 11 and all(c.isalnum() or c in '-_' for c in video_id)
    
    @staticmethod
    def get_video_info(video_id: str) -> Dict[str, str]:
        """
        Get video metadata information.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing video ID and basic info
        """
        try:
            # Validate video ID format
            if not YouTubeUtils.validate_video_id(video_id):
                return {
                    "title": "YouTube Video",
                    "video_id": video_id,
                    "error": "Invalid video ID format"
                }
            
            try:
                video = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                return {
                    "title": video.title or f"YouTube Video - {video_id}",
                    "author": video.author or "Unknown",
                    "length": str(video.length or 0),
                    "views": str(video.views or 0),
                    "description": video.description or "",
                    "video_id": video_id
                }
            except Exception:
                return {
                    "title": f"YouTube Video - {video_id}",
                    "author": "Unknown",
                    "length": "0",
                    "views": "0",
                    "description": "",
                    "video_id": video_id
                }
        except Exception as e:
            return {
                "title": "YouTube Video",
                "author": "Unknown",
                "length": "0",
                "views": "0",
                "description": "",
                "video_id": video_id,
                "error": str(e)
            }
    
    @staticmethod
    def get_transcript_segments(video_id: str, languages: list = ['en']) -> Optional[List[Dict[str, Any]]]:
        """
        Get video transcript as timestamped segments.

        Args:
            video_id: YouTube video ID
            languages: List of language codes to try

        Returns:
            List of transcript segment dicts with text, start, duration
        """
        try:
            api = YouTubeTranscriptApi()
            transcript_obj = api.fetch(
                video_id=video_id,
                languages=languages
            )

            segments = []
            for snippet in transcript_obj:
                text = " ".join(snippet.text.split())
                if not text:
                    continue
                segments.append({
                    "text": text,
                    "start": float(snippet.start),
                    "duration": float(snippet.duration)
                })

            return segments

        except Exception as e:
            print(f"Error fetching transcript: {str(e)}")
            return None

    @staticmethod
    def get_transcript(video_id: str, languages: list = ['en']) -> Optional[str]:
        """
        Get video transcript as a single string.

        Args:
            video_id: YouTube video ID
            languages: List of language codes to try

        Returns:
            Transcript text if available, None otherwise
        """
        segments = YouTubeUtils.get_transcript_segments(video_id, languages=languages)
        if not segments:
            return None
        return " ".join(segment["text"] for segment in segments)
    
    @staticmethod
    def get_available_transcripts(video_id: str) -> list:
        """
        Get list of available transcript languages.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of available language codes
        """
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            return list(transcript_list)
        except Exception as e:
            print(f"Error fetching available transcripts: {str(e)}")
            return []
