"""
YouTube ChatBot Module
Implements the conversational AI for answering questions about YouTube videos.
"""

from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
        
        self.text_processor = TextProcessor()
        self.vectorstore = None
        self.chain = None
        self.chat_history = []
        self.video_context = None
    
    def initialize_vectorstore(self, transcript: str, video_info: dict = None):
        """
        Initialize vector store with video transcript.
        
        Args:
            transcript: Video transcript text
            video_info: Optional video metadata
        """
        # Store video context
        self.video_context = video_info
        
        # Preprocess and chunk the transcript
        processed_text = self.text_processor.preprocess_text(transcript)
        text_chunks = self.text_processor.chunk_text(processed_text)
        
        if not text_chunks:
            raise ValueError("No text chunks created from transcript")
        
        # Create vector store
        self.vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=self.embeddings
        )
        
        # Create the retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an AI assistant that answers questions **strictly based on the transcript of a YouTube video**.

        Your goals:
        1. **Ground every answer in the provided transcript.**
        2. **Never guess or invent information.**
        3. If the transcript does not contain the answer, reply with:
        "I cannot find that information in the video."
        4. Keep answers **clear, concise, friendly, and human-like**.
        5. If helpful, you may quote short parts of the transcript verbatim (but never invent quotes).

        Rules:
        - Do NOT use outside knowledge.
        - Do NOT assume anything that is not explicitly in the transcript.
        - Do NOT summarize the entire transcript unless asked.
        - If the question is vague, answer using only what the transcript provides.
        - If multiple interpretations exist, choose the one most directly supported by the text.

        Context from the video transcript:
        {context}
        """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        
        # Create the chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.chat_history
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Reset chat history
        self.chat_history = []
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question about the video.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.chain:
            return {
                "answer": "Please load a video first before asking questions.",
                "sources": []
            }
        
        try:
            # Get response from the chain
            answer = self.chain.invoke(question)
            
            # Add to chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
            
            return {
                "answer": answer,
                "sources": []
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
