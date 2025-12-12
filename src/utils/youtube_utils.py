"""
YouTube Utilities Module
Handles YouTube video information extraction and transcript retrieval.
"""

from typing import Optional, Dict
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeUtils:
    """Utility class for YouTube video operations."""
    
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
            
            # Just return basic info without fetching metadata
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
    def get_transcript(video_id: str, languages: list = ['en']) -> Optional[str]:
        """
        Get video transcript.
        
        Args:
            video_id: YouTube video ID
            languages: List of language codes to try
            
        Returns:
            Transcript text if available, None otherwise
        """
        try:
            api = YouTubeTranscriptApi()
            transcript_obj = api.fetch(
                video_id=video_id, 
                languages=languages
            )
            
            # Combine all transcript segments
            # The transcript_obj is a FetchedTranscript object with snippets attribute
            transcript_text = " ".join([snippet.text for snippet in transcript_obj])
            return transcript_text
        
        except Exception as e:
            print(f"Error fetching transcript: {str(e)}")
            return None
    
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
