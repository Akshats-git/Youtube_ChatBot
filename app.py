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


def load_video(video_id: str, language: str = 'en'):
    """Load and process a YouTube video."""
    try:
        # Validate video ID
        if not YouTubeUtils.validate_video_id(video_id):
            st.error(f"Invalid YouTube video ID. Video IDs are 11 characters long.\nExample: aqz-KE-bpKQ")
            return False
        
        st.info("🔍 Validating video ID...")
        
        # Show loading message
        with st.spinner("📊 Loading video information..."):
            # Get video info
            video_info = YouTubeUtils.get_video_info(video_id)
            
            if "error" in video_info:
                st.info(f"⚠️ Could not fetch video metadata, but continuing with transcript...")
        
        with st.spinner(f"📥 Fetching video transcript in {language.upper()}... This may take a moment..."):
            # Get transcript with selected language
            transcript = YouTubeUtils.get_transcript(video_id, languages=[language])
            
            if not transcript:
                st.error(f"❌ Could not fetch transcript for this video in {language.upper()}.\n\nMake sure the video has captions available in the selected language.")
                return False
            
            st.success(f"✅ Transcript fetched successfully! ({len(transcript)} characters)")
        
        with st.spinner("🤖 Initializing AI chatbot..."):
            # Initialize chatbot
            chatbot = YouTubeChatBot()
            chatbot.initialize_vectorstore(transcript, video_info)
        
        # Update session state
        st.session_state.chatbot = chatbot
        st.session_state.video_loaded = True
        st.session_state.video_info = video_info
        st.session_state.current_video_id = video_id
        st.session_state.chat_history = []
        st.session_state.selected_language = language
        
        st.success("✅ Video loaded successfully! You can now ask questions about the video.")
        return True
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return False


def display_video_info():
    """Display video information in the sidebar."""
    if st.session_state.video_info:
        info = st.session_state.video_info
        st.sidebar.markdown("### Video Information")
        st.sidebar.markdown(f"**Video ID:** {info.get('video_id', 'N/A')}")
        
        if st.sidebar.button("Load New Video", use_container_width=True):
            st.session_state.video_loaded = False
            st.session_state.video_info = None
            st.session_state.chat_history = []
            st.session_state.chatbot = None
            st.rerun()


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
        
        # Add bot response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': answer
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
        1. Enter a YouTube video ID (11 characters)
        2. Select the transcript language
        3. Click 'Load Video'
        4. Start asking questions about the video content
        """)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This assistant uses AI to answer questions about YouTube videos 
        based on their transcripts. 
        """)
    
    # Main content area
    if not st.session_state.video_loaded:
        # Video input section
        st.markdown('<div class="section-header">Load Video</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            video_id = st.text_input(
                "Enter YouTube Video ID:",
                placeholder="aqz-KE-bpKQ (11 characters)",
                label_visibility="collapsed"
            )
        
        with col2:
            load_button = st.button("Load Video", type="primary", use_container_width=True)
        
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
        
        if load_button and video_id:
            load_video(video_id, language=selected_language_code)
            st.rerun()
        
        # Example section
        # st.markdown("<br>", unsafe_allow_html=True)
        # with st.expander("View Example Video IDs"):
        #     st.code("aqz-KE-bpKQ")
    
    else:
        # Display chat interface
        display_chat_interface()
        
        # Clear chat button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.reset_conversation()
                st.rerun()


if __name__ == "__main__":
    main()
