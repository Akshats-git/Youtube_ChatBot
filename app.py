"""
YouTube ChatBot — Streamlit Application
Main entry point for the YouTube video assistant.
"""

import traceback
from typing import List, Dict, Any

import streamlit as st

from src.chatbot import YouTubeChatBot
from src.utils.youtube_utils import YouTubeUtils, format_timestamp
from config.config import Config


st.set_page_config(
    page_title="YouTube ChatBot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { background-color: #ffffff; }

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
        color: #e5e7eb;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    .video-info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

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
        to   { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        border-radius: 12px 12px 4px 12px;
    }

    .user-message strong { color: #ffffff; font-weight: 600; }

    .bot-message {
        background-color: #f3f4f6;
        margin-right: 20%;
        border-radius: 12px 12px 12px 4px;
        border-left: 3px solid #667eea;
    }

    .bot-message strong { color: #667eea; font-weight: 600; }

    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f9fafb;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #374151;
    }

    .transcript-box {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 0.75rem;
        color: #f9fafb;
        max-height: 320px;
        overflow-y: auto;
        font-size: 0.92rem;
        line-height: 1.45;
        white-space: pre-wrap;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid #374151;
        background: #111827;
        padding: 0.4rem 0.6rem;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] div { color: #f3f4f6; }

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

    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .stAlert { border-radius: 8px; }
    .stCodeBlock { border-radius: 8px; background-color: #f3f4f6; }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def initialize_session_state() -> None:
    defaults = {
        "chatbot": None,
        "video_loaded": False,
        "video_info": None,
        "chat_history": [],
        "current_video_id": "",
        "selected_language": "en",
        "transcript_segments": [],
        "transcript_text": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video(video_input: str, language: str = "en", force_rebuild: bool = False) -> bool:
    """Validate, fetch, and index a YouTube video. Returns True on success."""
    video_id = YouTubeUtils.extract_video_id(video_input)
    if not video_id:
        st.error("Invalid YouTube URL or video ID. Please provide a valid link or 11-character ID.")
        return False

    try:
        with st.status("Loading video...", expanded=True) as status:
            status.update(label="Validating video ID", state="running")
            if not YouTubeUtils.validate_video_id(video_id):
                status.update(label="Invalid video ID", state="error")
                st.error("Invalid YouTube video ID. Example: aqz-KE-bpKQ")
                return False

            status.update(label="Fetching video metadata", state="running")
            video_info = YouTubeUtils.get_video_info(video_id)
            if "error" in video_info:
                st.info("Could not fetch full metadata; continuing with transcript only.")

            status.update(label=f"Fetching transcript ({language.upper()})", state="running")
            transcript_segments = YouTubeUtils.get_transcript_segments(
                video_id, languages=[language]
            )
            if not transcript_segments:
                status.update(label="Transcript unavailable", state="error")
                st.error(
                    f"No transcript found for this video in {language.upper()}. "
                    "Ensure captions are enabled for the selected language."
                )
                return False

            transcript_text = " ".join(s["text"] for s in transcript_segments)
            status.update(
                label=f"Transcript fetched ({len(transcript_text):,} characters)",
                state="complete",
            )

            status.update(label="Building RAG index", state="running")
            chatbot = YouTubeChatBot()
            chatbot.initialize_vectorstore(
                transcript_segments=transcript_segments,
                video_info=video_info,
                video_id=video_id,
                language=language,
                force_rebuild=force_rebuild,
            )
            status.update(label="Ready to chat", state="complete")

        st.session_state.chatbot = chatbot
        st.session_state.video_loaded = True
        st.session_state.video_info = video_info
        st.session_state.current_video_id = video_id
        st.session_state.chat_history = []
        st.session_state.selected_language = language
        st.session_state.transcript_segments = transcript_segments
        st.session_state.transcript_text = transcript_text

        st.success("Video loaded! Ask your first question below.")
        return True

    except Exception as exc:
        st.error(f"An error occurred: {exc}")
        st.error(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def display_video_info() -> None:
    info = st.session_state.video_info or {}
    st.sidebar.markdown("### Video Information")
    st.sidebar.markdown(f"**Title:** {info.get('title', 'N/A')}")
    st.sidebar.markdown(f"**Video ID:** {info.get('video_id', 'N/A')}")
    st.sidebar.markdown(f"**Language:** {st.session_state.selected_language.upper()}")

    if st.sidebar.button("Load New Video", use_container_width=True):
        for key in ("video_loaded", "video_info", "chat_history", "chatbot",
                    "transcript_segments", "transcript_text"):
            st.session_state[key] = (
                False if key == "video_loaded"
                else [] if key in ("chat_history", "transcript_segments")
                else "" if key == "transcript_text"
                else None
            )
        st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_chat_markdown(chat_history: list) -> str:
    """Render the conversation as a Markdown document for download."""
    lines = []
    for message in chat_history:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"**{role}:** {message.get('content', '')}")
        for source in message.get("sources", []):
            time_range = (
                f"{format_timestamp(source['start_time'])}"
                f"-{format_timestamp(source['end_time'])}"
            )
            snippet = source.get("text", "").strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            lines.append(f"- [{source['index']}] {time_range} {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_transcript_lines(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = start + float(seg.get("duration", 0.0))
        lines.append(f"[{format_timestamp(start)}-{format_timestamp(end)}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def display_summary_panel() -> None:
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
        st.markdown("\n".join(f"- {t}" for t in topics))


def display_transcript_panel() -> None:
    segments = st.session_state.get("transcript_segments", [])
    transcript_text = st.session_state.get("transcript_text", "")
    if not transcript_text and not segments:
        return

    st.markdown('<div class="section-header">Transcript</div>', unsafe_allow_html=True)
    timestamped = _build_transcript_lines(segments)
    content = timestamped or transcript_text

    col1, col2 = st.columns([3, 1])
    with col1:
        with st.expander("View transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{content}</div>',
                unsafe_allow_html=True,
            )
    with col2:
        st.download_button(
            "Download Transcript",
            data=content,
            file_name=f"transcript_{st.session_state.current_video_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def display_chat_interface() -> None:
    st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.info("No messages yet — ask a question below to start.")

    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message("user" if role == "user" else "assistant", avatar=avatar):
            st.markdown(message.get("content", ""))
            if role == "assistant":
                sources = message.get("sources", [])
                if sources:
                    with st.expander("Sources used for this reply", expanded=False):
                        for source in sources:
                            time_range = (
                                f"{format_timestamp(source['start_time'])}"
                                f"-{format_timestamp(source['end_time'])}"
                            )
                            snippet = source.get("text", "").strip().replace("\n", " ")
                            if len(snippet) > 300:
                                snippet = snippet[:300] + "..."
                            st.markdown(f"- [{source['index']}] {time_range} {snippet}")

    user_question = st.chat_input("Ask about the video")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.spinner("Thinking…"):
            response = st.session_state.chatbot.ask_question(user_question)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.get("answer", ""),
            "sources": response.get("sources", []),
        })
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LANGUAGE_OPTIONS: Dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh-Hans",
    "Chinese (Traditional)": "zh-Hant",
    "Hindi": "hi",
    "Arabic": "ar",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Filipino": "fil",
    "Polish": "pl",
    "Dutch": "nl",
}


def main() -> None:
    initialize_session_state()

    st.markdown(
        '<div class="main-header">YouTube Video Assistant</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Interact with YouTube videos using AI-powered conversations</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Settings")
        try:
            Config.validate()
            st.success("API Key Configured")
        except ValueError as exc:
            st.error(str(exc))
            st.info("Please set your OpenAI API key in the `.env` file.")
            st.stop()

        st.markdown("---")
        if st.session_state.video_loaded:
            display_video_info()

        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown(
            "1. Paste a YouTube URL or video ID\n"
            "2. Select the transcript language\n"
            "3. Click **Load Video** to build the index\n"
            "4. Ask questions about the video content"
        )
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "Uses retrieval-augmented generation (RAG) to answer questions "
            "grounded in YouTube transcripts."
        )

    if not st.session_state.video_loaded:
        st.markdown('<div class="section-header">Load Video</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            video_input = st.text_input(
                "YouTube URL or Video ID",
                placeholder="https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                label_visibility="collapsed",
            )
        with col2:
            load_button = st.button("Load Video", type="primary", use_container_width=True)

        force_rebuild = st.checkbox(
            "Rebuild index",
            value=False,
            help="Recreate the vector index even if a cached version exists.",
        )

        st.markdown("")
        selected_name = st.selectbox(
            "Transcript Language", options=list(LANGUAGE_OPTIONS.keys()), index=0
        )
        selected_code = LANGUAGE_OPTIONS[selected_name]

        if load_button and video_input:
            if load_video(video_input, language=selected_code, force_rebuild=force_rebuild):
                st.rerun()

    else:
        display_summary_panel()
        st.markdown("<br>", unsafe_allow_html=True)
        display_transcript_panel()
        st.markdown("<br>", unsafe_allow_html=True)
        display_chat_interface()

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, _ = st.columns([1, 2, 3])
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.reset_conversation()
                st.rerun()
        with col2:
            st.download_button(
                "Download Chat",
                data=build_chat_markdown(st.session_state.chat_history),
                file_name=f"chat_{st.session_state.current_video_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
