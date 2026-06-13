# YouTube ChatBot

A Streamlit web application that lets you have an AI-powered conversation with any YouTube video. Paste a URL, wait a few seconds for the index to build, then ask questions — the assistant answers using only the video transcript and cites exact timestamps so you can verify every claim.

---

## Features

- **Hybrid RAG retrieval** — combines FAISS vector search with BM25 lexical search for more precise context selection
- **Query expansion** — automatically rewrites and generates multiple search variants before retrieval
- **Optional context compression** — an LLM pass trims retrieved chunks to the most relevant sentences
- **Timestamped citations** — every answer links back to the exact segment of the video
- **Persistent index cache** — FAISS indexes are saved to disk so re-loading the same video is instant
- **Multi-language transcripts** — supports 20 languages including Spanish, French, German, Japanese, and more
- **Video summary** — automatically generates a summary and key topics on load
- **Conversation memory** — the model retains context across turns within a session
- **Chat export** — download the full conversation as Markdown

---

## Architecture

### Video loading pipeline

```
  YouTube URL / ID
        │
        ▼
┌───────────────────┐
│  URL parsing &    │  extract_video_id()
│  ID validation    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐        ┌──────────────────────┐
│  Fetch metadata   │──────▶ │  pytube (title,       │
│  (best-effort)    │        │  author, duration…)   │
└────────┬──────────┘        └──────────────────────┘
         │
         ▼
┌───────────────────┐        ┌──────────────────────┐
│  Fetch transcript │──────▶ │  youtube-transcript-  │
│  segments         │        │  api  {text,start,    │
└────────┬──────────┘        │        duration}      │
         │                   └──────────────────────┘
         ▼
┌───────────────────┐
│  Chunk & embed    │  RecursiveCharacterTextSplitter
│  (TextProcessor)  │  → timestamp metadata per chunk
└────────┬──────────┘
         │
         ▼
┌───────────────────┐        ┌──────────────────────┐
│  FAISS index      │        │  BM25 index           │
│  (vector store)   │        │  (lexical store)      │
│  saved to disk ◀──┼────────┼──▶ in-memory          │
└───────────────────┘        └──────────────────────┘
```

### Query answering pipeline

```
  User question
        │
        ▼
┌───────────────────┐
│  Small-talk       │──yes──▶  greeting response (no retrieval)
│  detection        │
└────────┬──────────┘
         │ no
         ▼
┌───────────────────┐
│  Query rewrite    │  "What did he say about X?"
│  (query LLM)      │──▶ "X explanation keywords"
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Multi-query      │  generates N variants of the
│  expansion        │  rewritten query
└────────┬──────────┘
         │  [q1, q2, q3 … qN]  (deduplicated)
         ▼
┌────────────────────────────────────────┐
│            Hybrid retrieval            │
│                                        │
│  ┌─────────────┐   ┌─────────────────┐ │
│  │ FAISS vector│   │   BM25 lexical   │ │
│  │  search ×N  │   │   search ×N      │ │
│  └──────┬──────┘   └───────┬─────────┘ │
│         │                  │            │
│         └────────┬─────────┘            │
│                  ▼                       │
│       Reciprocal-rank fusion             │
│       (merge & re-score candidates)      │
└──────────────────┬─────────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │  Compression      │  LLMChainExtractor trims each
        │  (optional)       │  chunk to the relevant sentences
        └────────┬──────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Context assembly │  top-K chunks formatted with
        │                  │  [idx] (HH:MM:SS-HH:MM:SS) text
        └────────┬──────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Answer LLM       │  responds strictly from context,
        │  + chat history   │  cites [1], [2] … source indices
        └────────┬──────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Citation filter  │  keeps only sources actually
        │                  │  referenced in the answer
        └────────┬──────────┘
                 │
                 ▼
          answer + sources
```

---

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Quickstart

```bash
# 1. Clone the repository
git clone <repo-url>
cd YouTube_ChatBot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=<your-key>

# 5. Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Configuration

All settings are controlled via `.env`. Copy `.env.example` as a starting point.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `MODEL_NAME` | `gpt-3.5-turbo` | Chat model used for answers and query rewriting |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding model for FAISS indexing |
| `TEMPERATURE` | `0.7` | Sampling temperature for the answer LLM |
| `QUERY_TEMPERATURE` | `0.0` | Temperature for query rewriting (deterministic) |
| `SUMMARY_TEMPERATURE` | `0.2` | Temperature for video summarisation |
| `MAX_TOKENS` | `500` | Max tokens per LLM call |
| `CHUNK_SIZE` | `1000` | Target character size per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `SNIPPET_MAX_CHARS` | `1200` | Max chars from a single chunk included in the prompt |
| `RETRIEVAL_K` | `6` | FAISS candidates retrieved per query |
| `BM25_K` | `6` | BM25 candidates retrieved per query |
| `FINAL_K` | `6` | Top-K documents sent to the LLM after score fusion |
| `MULTIQUERY_COUNT` | `3` | Number of additional query variants generated |
| `MAX_CONTEXT_CHARS` | `8000` | Total character budget for the context block |
| `ENABLE_QUERY_REWRITE` | `true` | Toggle query rewriting |
| `ENABLE_MULTIQUERY` | `true` | Toggle multi-query expansion |
| `ENABLE_COMPRESSION` | `true` | Toggle LLM-based context compression |
| `INDEX_DIR` | `data/indexes` | Directory for persisted FAISS indexes |
| `SUMMARY_MAX_CHUNKS` | `6` | Chunks sampled for the auto-summary |
| `SUMMARY_MAX_TOKENS` | `300` | Max tokens for the summary response |

---

## Usage

1. **Paste a URL** — any standard YouTube link (`watch?v=`, `youtu.be/`, `/shorts/`) or a bare 11-character video ID
2. **Select language** — choose the transcript language from the dropdown (defaults to English)
3. **Load Video** — the app fetches the transcript and builds the FAISS index (cached on disk after the first run)
4. **Ask questions** — type in the chat box; answers include `[1]`, `[2]` citations you can expand to see the source segment and timestamp
5. **Export** — download the transcript or the full conversation as Markdown

---

## How It Works

### Video loading

- The video ID is extracted from any supported URL format
- `pytube` fetches metadata (title, author, duration); if that fails, the app continues with transcript-only mode
- `youtube-transcript-api` downloads the selected language's captions as timestamped segments

### Indexing

- Segments are concatenated and split into overlapping chunks with `RecursiveCharacterTextSplitter`
- Each chunk is annotated with `start_time`/`end_time` derived from the original segment offsets
- Chunks are embedded and stored in a FAISS index that is saved to `data/indexes/` for reuse

### Question answering

- The user's question is rewritten and expanded into multiple search queries
- Each query is run against both the FAISS and BM25 indexes; results are merged by reciprocal-rank fusion
- Optionally, an LLM compresses the top candidates to the most relevant sentences
- The final context (with timestamps) is injected into the system prompt; the model answers and cites sources

---

## Technology Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| LLM & embeddings | OpenAI (`gpt-3.5-turbo`, `text-embedding-ada-002`) |
| Orchestration | LangChain |
| Vector store | FAISS (`faiss-cpu`) |
| Lexical retrieval | `rank_bm25` via `langchain-community` |
| Transcript extraction | `youtube-transcript-api` |
| Video metadata | `pytube` |
| Text splitting | `langchain-text-splitters` |

---

## Project Structure

```
YouTube_ChatBot/
├── app.py                      # Streamlit entry point
├── config/
│   ├── __init__.py
│   └── config.py               # Environment-driven configuration
├── src/
│   ├── __init__.py
│   ├── chatbot.py              # YouTubeChatBot — RAG pipeline
│   └── utils/
│       ├── __init__.py
│       ├── youtube_utils.py    # Video metadata, transcript fetching, timestamp formatting
│       └── text_processor.py  # Chunking & Document construction
├── data/
│   └── indexes/                # Cached FAISS indexes (git-ignored)
├── .env.example                # Environment variable template
├── requirements.txt
└── README.md
```

---

## Troubleshooting

**"Could not fetch transcript"**
The video either has no captions or the selected language is unavailable. Try enabling auto-generated captions on YouTube or selecting a different language.

**"API Key not set"**
Ensure `.env` exists and contains `OPENAI_API_KEY=<your-key>`. Restart the app after editing the file.

**Import errors**
Run `pip install -r requirements.txt` inside your virtual environment and confirm it is activated.

**`pytube` metadata errors**
`pytube` occasionally breaks when YouTube updates its web app. Metadata failures are non-fatal — the app will continue with the transcript. If needed, pin a specific `pytube` version known to work.

---

## Limitations

- Requires captions to be available on the video (auto-generated captions are supported)
- Answers are limited to information present in the transcript; the model will not speculate beyond the source material
- Context window constraints mean that very long or dense videos may not surface every relevant passage in a single query
- Video metadata (title, author, views) is best-effort — it depends on `pytube` being compatible with the current YouTube frontend

---

## License

MIT
