# 🎥 YouTube ChatBot

An intelligent conversational AI application that allows users to interact with YouTube videos through natural language questions. Simply provide a YouTube video URL, and the chatbot will analyze the video's transcript to answer your questions about the content.

## ✨ Features

- 🎬 **Easy Video Loading**: Just paste any YouTube video URL
- 💬 **Natural Conversations**: Ask questions in plain English about the video content
- 🧠 **AI-Powered**: Uses OpenAI's GPT models and embeddings for accurate responses
- 📝 **Transcript Analysis**: Automatically extracts and processes video transcripts
- 🎨 **Beautiful UI**: Clean and intuitive Streamlit interface
- 💾 **Conversation Memory**: Maintains context throughout your chat session
- 🔍 **Smart Search**: Uses vector embeddings to find relevant information

## 🏗️ Project Structure

```
YouTube_ChatBot/
├── app.py                          # Main Streamlit application
├── config/
│   ├── __init__.py
│   └── config.py                   # Configuration management
├── src/
│   ├── __init__.py
│   ├── chatbot.py                  # ChatBot logic with LangChain
│   └── utils/
│       ├── __init__.py
│       ├── youtube_utils.py        # YouTube video operations
│       └── text_processor.py       # Text chunking and processing
├── data/                           # Data storage (gitignored)
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd YouTube_ChatBot
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # Activate on Linux/Mac:
   source venv/bin/activate
   
   # Activate on Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your_actual_api_key_here
   ```

### Configuration

Edit the `.env` file with your settings:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - defaults are set
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🎮 Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL manually

3. **Use the chatbot:**
   - Paste a YouTube video URL in the input field
   - Click "Load Video" and wait for processing
   - Start asking questions about the video!
   - Example questions:
     - "What is the main topic of this video?"
     - "Can you summarize the key points?"
     - "What does the speaker say about [specific topic]?"

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - OpenAI GPT-3.5/4 for chat
  - OpenAI Embeddings for semantic search
  - LangChain for orchestration
  - FAISS for vector storage
- **YouTube Integration**:
  - youtube-transcript-api for transcript extraction
  - pytube for video metadata
- **Text Processing**: 
  - RecursiveCharacterTextSplitter for intelligent chunking
  - tiktoken for token counting

## 📝 How It Works

1. **Video Loading**: When you provide a YouTube URL, the app:
   - Extracts the video ID
   - Fetches video metadata (title, author, duration)
   - Downloads the transcript using YouTube's caption API

2. **Text Processing**: The transcript is:
   - Cleaned and preprocessed
   - Split into manageable chunks with overlap
   - Converted to vector embeddings

3. **Question Answering**: When you ask a question:
   - Your question is converted to an embedding
   - Similar chunks are retrieved from the vector store
   - The AI generates an answer based on relevant context
   - Conversation history is maintained for context

## 🔒 Privacy & Security

- Your OpenAI API key is stored locally in `.env` (gitignored)
- No video data or conversations are stored permanently
- All processing happens on your machine and OpenAI's servers
- No third-party data collection

## ⚠️ Limitations

- Only works with videos that have transcripts/captions available
- English transcripts work best (other languages supported if available)
- Very long videos (>2 hours) may take longer to process
- Answers are limited to what's in the transcript

## 🤝 Contributing

Contributions are welcome! Here are some ideas:
- Support for multiple languages
- Add support for video playlists
- Implement local LLM support
- Add export chat history feature
- Improve UI/UX

## 📄 License

This project is open source and available under the MIT License.

## 🐛 Troubleshooting

### "Could not fetch transcript"
- The video may not have captions available
- Try a different video with captions enabled

### "API Key not set"
- Make sure your `.env` file exists
- Check that `OPENAI_API_KEY` is set correctly
- Restart the application after setting the key

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment

### Rate limiting
- If you hit OpenAI rate limits, wait a few moments
- Consider upgrading your OpenAI plan for higher limits

## 📧 Support

For issues and questions:
- Check existing issues in the repository
- Create a new issue with detailed information
- Include error messages and steps to reproduce

## 🙏 Acknowledgments

- OpenAI for GPT and embedding models
- LangChain for the excellent framework
- Streamlit for the awesome UI framework
- The open-source community

---

**Made with ❤️ by Akshat**
