"""
YouTube ChatBot Streamlit Application
Main frontend application for the YouTube video chatbot.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import YouTubeChatBot
from src.utils.youtube_utils import YouTubeUtils
from config.config import Config


# Page configuration
st.set_page_config(
    page_title="YouTube ChatBot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Video info card */
    .video-info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        color: #1f2937;
        line-height: 1.7;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        border-radius: 12px 12px 4px 12px;
    }
    
    .user-message strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    .bot-message {
        background-color: #f3f4f6;
        margin-right: 20%;
        border-radius: 12px 12px 12px 4px;
        border-left: 3px solid #667eea;
    }
    
    .bot-message strong {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Input styling */
    .stTextInput input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.6rem;
        font-size: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px;
        background-color: #f3f4f6;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = ""
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'en'


def load_video(video_input: str, language: str = 'en', force_rebuild: bool = False):
    """Load and process a YouTube video."""
    try:
        video_id = YouTubeUtils.extract_video_id(video_input)
        if not video_id:
            st.error("Invalid YouTube URL or video ID. Please provide a valid link or 11-character ID.")
            return False

        with st.status("Loading video...", expanded=True) as status:
            status.update(label="Validating video ID", state="running")
            if not YouTubeUtils.validate_video_id(video_id):
                status.update(label="Invalid video ID", state="error")
                st.error("Invalid YouTube video ID. Video IDs are 11 characters long. Example: aqz-KE-bpKQ")
                return False

            status.update(label="Fetching video metadata", state="running")
            video_info = YouTubeUtils.get_video_info(video_id)
            if "error" in video_info:
                st.info("Could not fetch metadata, but continuing with transcript.")

            status.update(label=f"Fetching transcript ({language.upper()})", state="running")
            transcript_segments = YouTubeUtils.get_transcript_segments(video_id, languages=[language])

            if not transcript_segments:
                status.update(label="Transcript fetch failed", state="error")
                st.error(
                    f"Could not fetch transcript for this video in {language.upper()}. "
                    "Make sure the video has captions available in the selected language."
                )
                return False

            transcript_text = " ".join(segment["text"] for segment in transcript_segments)
            status.update(
                label=f"Transcript fetched ({len(transcript_text)} characters)",
                state="complete"
            )

            status.update(label="Building RAG index", state="running")
            chatbot = YouTubeChatBot()
            chatbot.initialize_vectorstore(
                transcript_segments=transcript_segments,
                video_info=video_info,
                video_id=video_id,
                language=language,
                force_rebuild=force_rebuild
            )

            status.update(label="RAG index ready", state="complete")
            status.update(label="Ready to chat", state="complete")

        # Update session state
        st.session_state.chatbot = chatbot
        st.session_state.video_loaded = True
        st.session_state.video_info = video_info
        st.session_state.current_video_id = video_id
        st.session_state.chat_history = []
        st.session_state.selected_language = language

        st.success("Video loaded successfully! You can now ask questions about the video.")
        return True

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return False


def display_video_info():
    """Display video information in the sidebar."""
    if st.session_state.video_info:
        info = st.session_state.video_info
        st.sidebar.markdown("### Video Information")
        st.sidebar.markdown(f"**Title:** {info.get('title', 'N/A')}")
        st.sidebar.markdown(f"**Video ID:** {info.get('video_id', 'N/A')}")
        st.sidebar.markdown(f"**Language:** {st.session_state.selected_language.upper()}")
        
        if st.sidebar.button("Load New Video", use_container_width=True):
            st.session_state.video_loaded = False
            st.session_state.video_info = None
            st.session_state.chat_history = []
            st.session_state.chatbot = None
            st.rerun()


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    total_seconds = int(max(seconds, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_chat_markdown(chat_history: list) -> str:
    """Build a markdown transcript of the chat conversation."""
    lines = []
    for message in chat_history:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"**{role}:** {message.get('content', '')}")
        sources = message.get("sources", [])
        if sources:
            lines.append("Sources:")
            for source in sources:
                time_range = f"{format_timestamp(source['start_time'])}-{format_timestamp(source['end_time'])}"
                snippet = source.get("text", "").strip().replace("\n", " ")
                snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
                lines.append(f"- [{source['index']}] {time_range} {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def display_summary_panel():
    """Display summary and key topics for the loaded video."""
    chatbot = st.session_state.chatbot
    if not chatbot:
        return

    summary = chatbot.get_video_summary()
    topics = chatbot.get_video_topics()

    if not summary and not topics:
        return

    st.markdown('<div class="section-header">Video Summary</div>', unsafe_allow_html=True)
    if summary:
        st.markdown(summary)
    if topics:
        st.markdown("**Key topics**")
        st.markdown("\n".join([f"- {topic}" for topic in topics]))


def display_chat_interface():
    """Display the chat interface."""
    st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You</strong><br>{message['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Assistant</strong><br>{message['content']}
                    </div>
                """, unsafe_allow_html=True)

                sources = message.get("sources", [])
                if sources:
                    with st.expander("Sources", expanded=False):
                        for source in sources:
                            time_range = (
                                f"{format_timestamp(source['start_time'])}"
                                f"-{format_timestamp(source['end_time'])}"
                            )
                            snippet = source.get("text", "").strip().replace("\n", " ")
                            snippet = snippet[:300] + ("..." if len(snippet) > 300 else "")
                            st.markdown(f"- [{source['index']}] {time_range} {snippet}")
    else:
        st.info("No messages yet. Ask a question below to start the conversation.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask your question",
            placeholder="What is this video about?",
            key="user_input",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button("Send Message", use_container_width=True, type="primary")
    
    if submit_button and user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.chatbot.ask_question(user_question)
            answer = response['answer']
            sources = response.get('sources', [])
        
        # Add bot response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': answer,
            'sources': sources
        })
        
        st.rerun()


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">YouTube Video Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interact with YouTube videos using AI-powered conversations</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Settings")
        
        # Check API key
        try:
            Config.validate()
            st.success("API Key Configured")
        except ValueError as e:
            st.error(str(e))
            st.info("Please set your OpenAI API key in the `.env` file.")
            st.stop()
        
        st.markdown("---")
        
        # Display video info if loadedZAmnSJ-GPZk
        if st.session_state.video_loaded:
            display_video_info()
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Paste a YouTube URL or video ID
        2. Select the transcript language
        3. Click 'Load Video' to build the RAG index
        4. Ask questions about the video content
        """)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This assistant uses retrieval-augmented generation (RAG) to answer
        questions grounded in YouTube transcripts.
        """)
    
    # Main content area
    if not st.session_state.video_loaded:
        # Video input section
        st.markdown('<div class="section-header">Load Video</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            video_input = st.text_input(
                "Enter YouTube URL or Video ID:",
                placeholder="https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                label_visibility="collapsed"
            )
        
        with col2:
            load_button = st.button("Load Video", type="primary", use_container_width=True)

        force_rebuild = st.checkbox(
            "Rebuild index",
            value=False,
            help="Recreate the vector index even if a cached version exists."
        )
        
        # Language selection
        st.markdown("")
        language_options = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Chinese (Simplified)': 'zh-Hans',
            'Chinese (Traditional)': 'zh-Hant',
            'Hindi': 'hi',
            'Arabic': 'ar',
            'Turkish': 'tr',
            'Vietnamese': 'vi',
            'Thai': 'th',
            'Indonesian': 'id',
            'Filipino': 'fil',
            'Polish': 'pl',
            'Dutch': 'nl'
        }
        
        selected_language_name = st.selectbox(
            "Transcript Language",
            options=list(language_options.keys()),
            index=0
        )
        selected_language_code = language_options[selected_language_name]
        
        if load_button and video_input:
            success = load_video(
                video_input,
                language=selected_language_code,
                force_rebuild=force_rebuild
            )
            if success:
                st.rerun()
        
        # Example section
        # st.markdown("<br>", unsafe_allow_html=True)
        # with st.expander("View Example Video IDs"):
        #     st.code("aqz-KE-bpKQ")
    
    else:
        display_summary_panel()
        st.markdown("<br>", unsafe_allow_html=True)
        # Display chat interface
        display_chat_interface()
        
        # Clear chat button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 3])
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.reset_conversation()
                st.rerun()
        with col2:
            chat_markdown = build_chat_markdown(st.session_state.chat_history)
            st.download_button(
                "Download Chat",
                data=chat_markdown,
                file_name=f"chat_{st.session_state.current_video_id}.md",
                mime="text/markdown",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
