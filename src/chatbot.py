"""
YouTube ChatBot
Conversational AI for answering questions grounded in YouTube transcripts.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config.config import Config
from src.utils.text_processor import TextProcessor
from src.utils.youtube_utils import format_timestamp

try:
    from langchain.retrievers.document_compressors import LLMChainExtractor as _LLMChainExtractor
    _COMPRESSION_AVAILABLE = True
except ImportError:
    _COMPRESSION_AVAILABLE = False


class YouTubeChatBot:
    """ChatBot for conversational Q&A over YouTube video transcripts."""

    def __init__(self):
        Config.validate()

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.EMBEDDING_MODEL,
        )
        self.llm = ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )
        self.query_llm = ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.QUERY_TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )
        self.summary_llm = ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.SUMMARY_TEMPERATURE,
            max_tokens=Config.SUMMARY_MAX_TOKENS,
        )

        self.text_processor = TextProcessor()
        self.vectorstore: Optional[FAISS] = None
        self.vector_retriever = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.documents: List[Document] = []
        self.answer_chain = None
        self.chat_history: List = []
        self.video_context: Dict[str, Any] = {}
        self.video_id: Optional[str] = None
        self.language: str = "en"
        self.video_summary: str = ""
        self.video_topics: List[str] = []
        self.index_metadata: Dict[str, Any] = {}
        self.compressor = None

        self.answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an AI assistant that answers questions strictly based on a YouTube transcript.\n\n"
                "Rules:\n"
                "- Use ONLY the provided sources below.\n"
                "- If the answer is not in the sources, reply with: "
                "\"I cannot find that information in the video.\"\n"
                "- Keep answers concise and helpful.\n"
                "- Cite sources using bracketed numbers like [1], [2] that match the sources list.\n\n"
                "Sources:\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        self.query_rewrite_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Rewrite the question into a short, keyword-focused search query for transcript retrieval.",
            ),
            ("human", "{question}"),
        ])

        self.multiquery_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Generate {n} diverse search queries to retrieve relevant transcript passages. "
                "Return one query per line without numbering.",
            ),
            ("human", "{question}"),
        ])

        self.summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Summarize the transcript excerpts. Provide:\n"
                "Summary: 2-4 sentences.\n"
                "Topics: 5-8 bullet points.",
            ),
            ("human", "{context}"),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_vectorstore(
        self,
        transcript_segments: List[Dict[str, Any]],
        video_info: dict,
        video_id: str,
        language: str,
        force_rebuild: bool = False,
    ) -> None:
        """
        Build or load the FAISS vector index for a video.

        Args:
            transcript_segments: Timestamped transcript segments.
            video_info: Video metadata dict.
            video_id: YouTube video ID.
            language: BCP-47 transcript language code.
            force_rebuild: Ignore cached index and rebuild from scratch.
        """
        self.video_context = video_info or {}
        self.video_id = video_id
        self.language = language

        index_path = self._get_index_path(video_id, language)
        if index_path.exists() and not force_rebuild:
            if self._load_vectorstore(index_path):
                self._build_retrievers()
                self._build_answer_chain()
                self._generate_video_summary()
                self.chat_history = []
                return

        documents = self.text_processor.build_documents_from_segments(
            transcript_segments,
            video_id=video_id,
            language=language,
            video_title=video_info.get("title", "") if video_info else "",
        )

        if not documents:
            raise ValueError("No transcript documents were created for indexing.")

        self.documents = documents
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self._save_vectorstore(index_path)
        self._build_retrievers()
        self._build_answer_chain()
        self._generate_video_summary()
        self.chat_history = []

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about the loaded video.

        Args:
            question: User's question.

        Returns:
            Dict with ``answer`` (str) and ``sources`` (list of source dicts).
        """
        if not self.answer_chain:
            return {
                "answer": "Please load a video before asking questions.",
                "sources": [],
            }

        try:
            if self._is_small_talk(question):
                answer = (
                    "Hello! Ask me anything about this video's content and I will "
                    "answer using the transcript."
                )
                self.chat_history.append(HumanMessage(content=question))
                self.chat_history.append(AIMessage(content=answer))
                return {"answer": answer, "sources": []}

            search_query = self._rewrite_query(question)
            queries = self._dedupe_queries(
                [question, search_query] + self._generate_multi_queries(search_query)
            )

            retrieved_docs = self._retrieve_documents(queries)
            candidate_docs = retrieved_docs[: Config.FINAL_K * 2]
            compressed_docs = self._compress_docs(question, candidate_docs)
            context, sources = self._format_context(compressed_docs)

            if not context:
                answer = "I cannot find that information in the video."
                sources = []
            else:
                answer = self.answer_chain.invoke({
                    "question": question,
                    "context": context,
                    "chat_history": self.chat_history,
                })
                sources = self._filter_sources_by_citation(answer, sources)

            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
            return {"answer": answer, "sources": sources}

        except Exception as exc:
            return {"answer": f"An error occurred: {exc}", "sources": []}

    def get_video_summary(self) -> str:
        return self.video_summary

    def get_video_topics(self) -> List[str]:
        return self.video_topics

    def reset_conversation(self) -> None:
        """Clear the in-memory conversation history."""
        self.chat_history = []

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _get_index_path(self, video_id: str, language: str) -> Path:
        safe_id = f"{video_id}_{language}".replace("/", "_")
        return Path(Config.INDEX_DIR) / safe_id

    def _save_vectorstore(self, index_path: Path) -> None:
        if not self.vectorstore:
            return
        index_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(index_path))
        metadata = {
            "video_id": self.video_id,
            "language": self.language,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "doc_count": len(self.documents),
            "model": Config.MODEL_NAME,
        }
        (index_path / "index_meta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self.index_metadata = metadata

    def _load_vectorstore(self, index_path: Path) -> bool:
        try:
            meta_path = index_path / "index_meta.json"
            metadata: Dict[str, Any] = {}
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if not self._metadata_is_compatible(metadata):
                    return False

            self.vectorstore = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            docstore = getattr(self.vectorstore, "docstore", None)
            self.documents = (
                list(docstore._dict.values())
                if docstore and hasattr(docstore, "_dict")
                else []
            )
            self.index_metadata = metadata
            return True
        except Exception:
            return False

    def _metadata_is_compatible(self, metadata: Dict[str, Any]) -> bool:
        if not metadata:
            return True
        return (
            metadata.get("embedding_model") == Config.EMBEDDING_MODEL
            and metadata.get("chunk_size") == Config.CHUNK_SIZE
            and metadata.get("chunk_overlap") == Config.CHUNK_OVERLAP
            and metadata.get("language") == self.language
            and metadata.get("video_id") == self.video_id
        )

    # ------------------------------------------------------------------
    # Retriever / chain setup
    # ------------------------------------------------------------------

    def _build_retrievers(self) -> None:
        if not self.vectorstore:
            return
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        if self.documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = Config.BM25_K
        else:
            self.bm25_retriever = None

    def _build_answer_chain(self) -> None:
        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _generate_video_summary(self) -> None:
        if not self.documents:
            self.video_summary, self.video_topics = "", []
            return

        selected = sorted(
            self.documents, key=lambda d: d.metadata.get("chunk_id", 0)
        )[: Config.SUMMARY_MAX_CHUNKS]
        context = "\n\n".join(d.page_content for d in selected).strip()

        if not context:
            self.video_summary, self.video_topics = "", []
            return

        chain = self.summary_prompt | self.summary_llm | StrOutputParser()
        self.video_summary, self.video_topics = self._parse_summary(
            chain.invoke({"context": context})
        )

    def _parse_summary(self, output: str) -> Tuple[str, List[str]]:
        summary = ""
        topics: List[str] = []
        for line in output.splitlines():
            clean = line.strip()
            if not clean:
                continue
            lower = clean.lower()
            if lower.startswith("summary") and ":" in clean:
                summary = clean.split(":", 1)[1].strip()
            elif lower.startswith("topics"):
                continue
            elif clean.startswith(("-", "*")):
                topic = clean.lstrip("-* ").strip()
                if topic:
                    topics.append(topic)

        if not summary:
            summary = output.strip().split("\n", 1)[0].strip()

        return summary, topics

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def _rewrite_query(self, question: str) -> str:
        if not Config.ENABLE_QUERY_REWRITE:
            return question
        chain = self.query_rewrite_prompt | self.query_llm | StrOutputParser()
        return chain.invoke({"question": question}).strip() or question

    def _generate_multi_queries(self, question: str) -> List[str]:
        if not Config.ENABLE_MULTIQUERY or Config.MULTIQUERY_COUNT <= 0:
            return [question]
        chain = self.multiquery_prompt | self.query_llm | StrOutputParser()
        output = chain.invoke({"question": question, "n": Config.MULTIQUERY_COUNT})
        queries = [
            line.strip().lstrip("- ")
            for line in output.splitlines()
            if line.strip()
        ]
        return self._dedupe_queries([question] + queries)

    def _dedupe_queries(self, queries: List[str]) -> List[str]:
        seen: set = set()
        unique: List[str] = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(q.strip())
        return unique

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve_documents(self, queries: List[str]) -> List[Document]:
        if not self.vectorstore:
            return []

        candidates: Dict[str, Dict[str, Any]] = {}

        for query in queries:
            for doc, score in self.vectorstore.similarity_search_with_score(
                query, k=Config.RETRIEVAL_K
            ):
                # Convert L2 distance to a 0-1 similarity score
                self._merge_candidate(candidates, doc, 1.0 / (1.0 + float(score)), "vector")

            for rank, doc in enumerate(self._bm25_search(query)):
                # Rank-based reciprocal score
                self._merge_candidate(candidates, doc, 1.0 / float(rank + 1), "bm25")

        return [
            item["doc"]
            for item in sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        ]

    def _bm25_search(self, query: str) -> List[Document]:
        if not self.bm25_retriever:
            return []
        return self.bm25_retriever.invoke(query)

    def _merge_candidate(
        self,
        candidates: Dict[str, Dict[str, Any]],
        doc: Document,
        score: float,
        method: str,
    ) -> None:
        key = str(doc.metadata.get("chunk_id", "")) or doc.page_content[:40]
        existing = candidates.get(key)
        if existing:
            best = max(existing["score"], score)
            existing["score"] = best
            existing["doc"].metadata["score"] = best
            methods: set = existing["methods"]
            methods.add(method)
            existing["doc"].metadata["retrieval_methods"] = sorted(methods)
        else:
            doc.metadata["score"] = score
            doc.metadata["retrieval_methods"] = [method]
            candidates[key] = {"doc": doc, "score": score, "methods": {method}}

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def _compress_docs(self, question: str, docs: List[Document]) -> List[Document]:
        if not Config.ENABLE_COMPRESSION or not docs or not _COMPRESSION_AVAILABLE:
            return docs
        try:
            if not self.compressor:
                self.compressor = _LLMChainExtractor.from_llm(self.llm)
            return self.compressor.compress_documents(docs, question)
        except Exception:
            return docs

    def _format_context(self, docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        total_chars = 0

        for idx, doc in enumerate(docs[: Config.FINAL_K], start=1):
            start_time = float(doc.metadata.get("start_time", 0.0))
            end_time = float(doc.metadata.get("end_time", start_time))
            snippet = doc.page_content.strip()
            if len(snippet) > Config.SNIPPET_MAX_CHARS:
                snippet = snippet[: Config.SNIPPET_MAX_CHARS].rsplit(" ", 1)[0] + "..."

            entry = f"[{idx}] ({format_timestamp(start_time)}-{format_timestamp(end_time)}) {snippet}"
            if total_chars + len(entry) > Config.MAX_CONTEXT_CHARS:
                break

            parts.append(entry)
            total_chars += len(entry)
            sources.append({
                "index": idx,
                "chunk_id": doc.metadata.get("chunk_id"),
                "start_time": start_time,
                "end_time": end_time,
                "text": doc.page_content.strip(),
                "score": doc.metadata.get("score", 0.0),
                "source": doc.metadata.get("source", "transcript"),
            })

        return "\n\n".join(parts), sources

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_small_talk(self, question: str) -> bool:
        normalized = question.strip().lower()
        if not normalized:
            return True

        greetings = {
            "hi", "hello", "hey", "yo", "hola",
            "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "whats up",
        }
        if normalized in greetings:
            return True

        tokens = normalized.split()
        return len(tokens) <= 2 and any(t in normalized for t in ("hi", "hello", "hey"))

    def _filter_sources_by_citation(
        self, answer: str, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not sources or not answer:
            return []
        cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer) if m.isdigit()}
        if not cited:
            return []
        return [s for s in sources if int(s.get("index", -1)) in cited]
