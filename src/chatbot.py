"""
YouTube ChatBot Module
Implements the conversational AI for answering questions about YouTube videos.
"""

import importlib
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


class YouTubeChatBot:
    """ChatBot for interacting with YouTube video content."""
    
    def __init__(self):
        """Initialize the ChatBot with necessary components."""
        Config.validate()
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.EMBEDDING_MODEL
        )
        
        self.llm = ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )

        self.query_llm = ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.QUERY_TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )

        self.summary_llm = ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=0.2,
            max_tokens=Config.SUMMARY_MAX_TOKENS
        )
        
        self.text_processor = TextProcessor()
        self.vectorstore = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.documents: List[Document] = []
        self.answer_chain = None
        self.chat_history = []
        self.video_context = None
        self.video_id: Optional[str] = None
        self.language: str = "en"
        self.video_summary = ""
        self.video_topics: List[str] = []
        self.index_metadata: Dict[str, Any] = {}
        self.compressor = None

        self.answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an AI assistant that answers questions strictly based on a YouTube transcript.

Rules:
- Use ONLY the provided sources below.
- If the answer is not in the sources, reply with: "I cannot find that information in the video."
- Keep answers concise and helpful.
- Cite sources using bracketed numbers like [1], [2] that match the sources list.

Sources:
{context}
"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        self.query_rewrite_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Rewrite the question into a short, keyword-focused search query for transcript retrieval."
            ),
            ("human", "{question}")
        ])

        self.multiquery_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Generate {n} diverse search queries to retrieve relevant transcript passages. "
                "Return one query per line without numbering."
            ),
            ("human", "{question}")
        ])

        self.summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Summarize the transcript excerpts. Provide:\n"
                "Summary: 2-4 sentences.\n"
                "Topics: 5-8 bullet points."
            ),
            ("human", "{context}")
        ])
    
    def initialize_vectorstore(
        self,
        transcript_segments: List[Dict[str, Any]],
        video_info: dict,
        video_id: str,
        language: str,
        force_rebuild: bool = False
    ):
        """
        Initialize vector store with video transcript segments.

        Args:
            transcript_segments: Transcript segments with timestamps
            video_info: Video metadata
            video_id: YouTube video ID
            language: Transcript language code
            force_rebuild: Force rebuilding index even if cached
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
            video_title=video_info.get("title", "") if video_info else ""
        )

        if not documents:
            raise ValueError("No transcript documents were created for indexing")

        self.documents = documents
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self._save_vectorstore(index_path)
        self._build_retrievers()
        self._build_answer_chain()
        self._generate_video_summary()
        self.chat_history = []

    def _get_index_path(self, video_id: str, language: str) -> Path:
        safe_id = f"{video_id}_{language}".replace("/", "_")
        return Path(Config.INDEX_DIR) / safe_id

    def _save_vectorstore(self, index_path: Path) -> None:
        index_path.mkdir(parents=True, exist_ok=True)
        if not self.vectorstore:
            return

        self.vectorstore.save_local(str(index_path))
        metadata = {
            "video_id": self.video_id,
            "language": self.language,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "doc_count": len(self.documents),
            "model": Config.MODEL_NAME
        }
        meta_path = index_path / "index_meta.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.index_metadata = metadata

    def _load_vectorstore(self, index_path: Path) -> bool:
        try:
            meta_path = index_path / "index_meta.json"
            metadata = {}
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if not self._metadata_is_compatible(metadata):
                    return False

            self.vectorstore = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            docstore = getattr(self.vectorstore, "docstore", None)
            if docstore and hasattr(docstore, "_dict"):
                self.documents = list(docstore._dict.values())
            else:
                self.documents = []

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

    def _generate_video_summary(self) -> None:
        if not self.documents:
            self.video_summary = ""
            self.video_topics = []
            return

        selected_docs = sorted(
            self.documents,
            key=lambda doc: doc.metadata.get("chunk_id", 0)
        )[: Config.SUMMARY_MAX_CHUNKS]
        context = "\n\n".join(doc.page_content for doc in selected_docs)

        if not context.strip():
            self.video_summary = ""
            self.video_topics = []
            return

        summary_chain = self.summary_prompt | self.summary_llm | StrOutputParser()
        output = summary_chain.invoke({"context": context})
        self.video_summary, self.video_topics = self._parse_summary(output)

    def _parse_summary(self, output: str) -> Tuple[str, List[str]]:
        summary = ""
        topics: List[str] = []
        for line in output.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            lower = clean_line.lower()
            if lower.startswith("summary") and ":" in clean_line:
                summary = clean_line.split(":", 1)[1].strip()
                continue
            if lower.startswith("topics"):
                continue
            if clean_line.startswith(("-", "*")):
                topic = clean_line.lstrip("-* ").strip()
                if topic:
                    topics.append(topic)

        if not summary:
            summary = output.strip().split("\n", 1)[0].strip()

        return summary, topics

    def _rewrite_query(self, question: str) -> str:
        if not Config.ENABLE_QUERY_REWRITE:
            return question

        rewrite_chain = self.query_rewrite_prompt | self.query_llm | StrOutputParser()
        rewritten = rewrite_chain.invoke({"question": question}).strip()
        return rewritten or question

    def _generate_multi_queries(self, question: str) -> List[str]:
        if not Config.ENABLE_MULTIQUERY or Config.MULTIQUERY_COUNT <= 0:
            return [question]

        multi_chain = self.multiquery_prompt | self.query_llm | StrOutputParser()
        output = multi_chain.invoke({
            "question": question,
            "n": Config.MULTIQUERY_COUNT
        })

        queries = [
            line.strip().lstrip("- ")
            for line in output.splitlines()
            if line.strip()
        ]

        return self._dedupe_queries([question] + queries)

    def _dedupe_queries(self, queries: List[str]) -> List[str]:
        seen = set()
        unique_queries = []
        for query in queries:
            normalized = query.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_queries.append(query.strip())
        return unique_queries

    def _retrieve_documents(self, queries: List[str]) -> List[Document]:
        candidates: Dict[str, Dict[str, Any]] = {}

        if not self.vectorstore:
            return []

        for query in queries:
            vector_hits = self.vectorstore.similarity_search_with_score(
                query,
                k=Config.RETRIEVAL_K
            )
            for doc, score in vector_hits:
                normalized_score = 1.0 / (1.0 + float(score))
                self._merge_candidate(candidates, doc, normalized_score, "vector")

            if self.bm25_retriever:
                bm25_hits = self._bm25_search(query)
                for rank, doc in enumerate(bm25_hits):
                    normalized_score = 1.0 / float(rank + 1)
                    self._merge_candidate(candidates, doc, normalized_score, "bm25")

        sorted_candidates = sorted(
            candidates.values(),
            key=lambda item: item["score"],
            reverse=True
        )

        return [item["doc"] for item in sorted_candidates]

    def _bm25_search(self, query: str) -> List[Document]:
        if not self.bm25_retriever:
            return []

        if hasattr(self.bm25_retriever, "get_relevant_documents"):
            return self.bm25_retriever.get_relevant_documents(query)
        if hasattr(self.bm25_retriever, "invoke"):
            return self.bm25_retriever.invoke(query)
        if callable(self.bm25_retriever):
            return self.bm25_retriever(query)
        return []

    def _merge_candidate(
        self,
        candidates: Dict[str, Dict[str, Any]],
        doc: Document,
        score: float,
        method: str
    ) -> None:
        key = str(doc.metadata.get("chunk_id", "")) or doc.page_content[:40]
        existing = candidates.get(key)
        if existing:
            best_score = max(existing["score"], score)
            existing["score"] = best_score
            existing["doc"].metadata["score"] = best_score
            existing_methods = existing.get("methods", set())
            existing_methods.add(method)
            existing["methods"] = existing_methods
            existing["doc"].metadata["retrieval_methods"] = sorted(existing_methods)
        else:
            doc.metadata["score"] = score
            doc.metadata["retrieval_methods"] = [method]
            candidates[key] = {
                "doc": doc,
                "score": score,
                "methods": {method}
            }

    def _compress_docs(self, question: str, docs: List[Document]) -> List[Document]:
        if not Config.ENABLE_COMPRESSION or not docs:
            return docs

        try:
            if not self.compressor:
                module = importlib.import_module("langchain.retrievers.document_compressors")
                extractor = getattr(module, "LLMChainExtractor", None)
                if not extractor:
                    return docs
                self.compressor = extractor.from_llm(self.llm)
            return self.compressor.compress_documents(docs, question)
        except Exception:
            return docs

    def _format_context(self, docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        total_chars = 0

        for idx, doc in enumerate(docs[: Config.FINAL_K], start=1):
            start_time = float(doc.metadata.get("start_time", 0.0))
            end_time = float(doc.metadata.get("end_time", start_time))
            snippet = doc.page_content.strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200].rsplit(" ", 1)[0] + "..."

            entry = f"[{idx}] ({self._format_timestamp(start_time)}-{self._format_timestamp(end_time)}) {snippet}"
            if total_chars + len(entry) > Config.MAX_CONTEXT_CHARS:
                break

            context_parts.append(entry)
            total_chars += len(entry)

            sources.append({
                "index": idx,
                "chunk_id": doc.metadata.get("chunk_id"),
                "start_time": start_time,
                "end_time": end_time,
                "text": doc.page_content.strip(),
                "score": doc.metadata.get("score", 0.0),
                "source": doc.metadata.get("source", "transcript")
            })

        return "\n\n".join(context_parts), sources

    def _format_timestamp(self, seconds: float) -> str:
        total_seconds = int(max(seconds, 0))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def get_video_summary(self) -> str:
        return self.video_summary

    def get_video_topics(self) -> List[str]:
        return self.video_topics

    def _is_small_talk(self, question: str) -> bool:
        """Detect greeting/chitchat prompts that should not trigger transcript retrieval."""
        normalized = question.strip().lower()
        if not normalized:
            return True

        small_talk_patterns = {
            "hi", "hello", "hey", "yo", "hola",
            "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "whats up"
        }

        if normalized in small_talk_patterns:
            return True

        if len(normalized.split()) <= 2 and any(token in normalized for token in ["hi", "hello", "hey"]):
            return True

        return False

    def _filter_sources_by_citation(self, answer: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return only sources that are explicitly cited in the answer text."""
        if not sources or not answer:
            return []

        cited_indices = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
        if not cited_indices:
            return []

        return [source for source in sources if int(source.get("index", -1)) in cited_indices]
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question about the video.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.answer_chain:
            return {
                "answer": "Please load a video first before asking questions.",
                "sources": []
            }
        
        try:
            if self._is_small_talk(question):
                answer = (
                    "Hello! Ask me anything about this video's content, and I will answer "
                    "using the transcript."
                )
                self.chat_history.append(HumanMessage(content=question))
                self.chat_history.append(AIMessage(content=answer))
                return {
                    "answer": answer,
                    "sources": []
                }

            search_query = self._rewrite_query(question)
            queries = self._dedupe_queries(
                [question, search_query] + self._generate_multi_queries(search_query)
            )

            retrieved_docs = self._retrieve_documents(queries)
            candidate_docs = retrieved_docs[: max(Config.FINAL_K * 2, Config.FINAL_K)]
            compressed_docs = self._compress_docs(question, candidate_docs)
            context, sources = self._format_context(compressed_docs)

            if not context:
                answer = "I cannot find that information in the video."
                sources = []
            else:
                answer = self.answer_chain.invoke({
                    "question": question,
                    "context": context,
                    "chat_history": self.chat_history
                })
                sources = self._filter_sources_by_citation(answer, sources)
            
            # Add to chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
            
            return {
                "answer": answer,
                "sources": sources
            }
        
        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": []
            }
    
    def reset_conversation(self):
        """Reset the conversation memory."""
        self.chat_history = []
    
    def get_conversation_history(self) -> List[tuple]:
        """
        Get the conversation history.
        
        Returns:
            List of (question, answer) tuples
        """
        history = []
        for i in range(0, len(self.chat_history), 2):
            if i + 1 < len(self.chat_history):
                history.append((
                    self.chat_history[i].content,
                    self.chat_history[i + 1].content
                ))
        
        return history
