"""
Enhanced Basketball RAG System

This module combines the best elements of both RAG implementations with improvements:
1. Modern vector embeddings with FAISS indexing
2. Support for both structured and unstructured data
3. Intelligent query routing
4. Advanced chunking strategies
5. Citation and source tracking
6. Improved error handling and logging
7. Token usage tracking
8. Conversation history management
"""

import os
import pandas as pd
import numpy as np
import logging
import json
from typing import Union, List, Tuple, Dict, Any, Optional
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import tiktoken
import dotenv
from openai import OpenAI
from dataclasses import dataclass

# Initialize environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basketball_rag.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class RetrievalResult:
    """Structure for storing retrieval results"""
    content: str
    metadata: Dict[str, Any]
    score: float

class BasketballRAG:
    """
    Enhanced RAG system for Basketball analytics
    Supporting both structured and unstructured data with advanced retrieval
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the RAG system with models and data structures
        
        Args:
            openai_api_key: API key for OpenAI, uses env var if None
            embedding_model: Name of sentence-transformers model to use
            llm_model: Name of OpenAI model to use
        """
        # Load API key from environment if not provided
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Provide it directly or set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize data containers
        self.structured_data = None
        self.unstructured_chunks = []
        self.unstructured_metadata = []
        
        # Load embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize FAISS index (will be created when data is loaded)
        self.faiss_index = None
        
        # Set LLM model
        self.llm_model = llm_model
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.encoding_for_model(llm_model) if "gpt" in llm_model else None
        
        # Query history
        self.query_history = []
        
        # Chat history for maintaining context
        self.chat_history = []
        
        logging.info(f"BasketballRAG system initialized with {embedding_model} embeddings and {llm_model}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens for a given text using the appropriate tokenizer"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation (1 token ≈ 4 characters for English text)
            return len(text) // 4

    def load_structured_data(self, file_path: str) -> None:
        """
        Load structured data from CSV or Excel file
        
        Args:
            file_path: Path to the CSV or Excel file
        """
        try:
            if file_path.endswith('.csv'):
                self.structured_data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.structured_data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
            
            logging.info(f"Successfully loaded structured data from {file_path} with {len(self.structured_data)} rows")
        except Exception as e:
            logging.error(f"Error loading structured data: {str(e)}")
            raise

    def _create_chunks_from_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Tuple[str, Dict]]:
        """
        Create chunks from text with overlap for better retrieval
        
        Args:
            text: The text to chunk
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of (chunk, metadata) tuples
        """
        if not text or len(text) <= chunk_size:
            return [(text, {"source": "unstructured", "chunk_id": 0})]
            
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            
            # If we're not at the end of the text, try to find a good breakpoint
            if end < len(text):
                # Look for paragraph break, sentence break, or at least a space
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break
                else:
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 1  # Include the period
                    else:
                        space_break = text.rfind(" ", start, end)
                        if space_break != -1:
                            end = space_break
            
            # Extract the chunk and its metadata
            chunk = text[start:end].strip()
            metadata = {
                "source": "unstructured",
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": end
            }
            
            chunks.append((chunk, metadata))
            chunk_id += 1
            
            # Move the start position for the next chunk, considering overlap
            start = end - chunk_overlap if end < len(text) else len(text)
        
        return chunks

    def load_unstructured_data(self, text_data: Union[str, List[str]], chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Load and chunk unstructured text data
        
        Args:
            text_data: String or list of strings containing text data
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        try:
            # Convert to list if it's a single string
            if isinstance(text_data, str):
                text_data = [text_data]
                
            # Clear existing unstructured data
            self.unstructured_chunks = []
            self.unstructured_metadata = []
            
            # Process each text document
            doc_id = 0
            for text in text_data:
                # Add document metadata to each chunk
                chunks_with_metadata = self._create_chunks_from_text(text, chunk_size, chunk_overlap)
                
                for chunk_text, metadata in chunks_with_metadata:
                    # Add document ID to metadata
                    metadata["doc_id"] = doc_id
                    
                    self.unstructured_chunks.append(chunk_text)
                    self.unstructured_metadata.append(metadata)
                
                doc_id += 1
            
            # Build FAISS index for the chunks
            self._build_faiss_index()
            
            logging.info(f"Successfully loaded and chunked {len(text_data)} documents into {len(self.unstructured_chunks)} chunks")
        except Exception as e:
            logging.error(f"Error loading unstructured data: {str(e)}")
            raise
    
    def _build_faiss_index(self) -> None:
        """Build a FAISS index for unstructured text chunks"""
        if not self.unstructured_chunks:
            logging.warning("No unstructured chunks to index")
            return
            
        try:
            # Get embeddings for all chunks
            embeddings = self.embedding_model.encode(
                self.unstructured_chunks, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Normalize embeddings
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            
            # Create the index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Using inner product (cosine similarity)
            self.faiss_index.add(embeddings)
            
            logging.info(f"FAISS index built with {len(self.unstructured_chunks)} chunks and dimension {dimension}")
        except Exception as e:
            logging.error(f"Error building FAISS index: {str(e)}")
            raise
    
    def _retrieve_from_structured(self, query: str, filter_params: Dict = None) -> List[RetrievalResult]:
        """
        Retrieve information from structured data based on the query
        
        Args:
            query: User query
            filter_params: Dict of column-value pairs to filter the data
            
        Returns:
            List of RetrievalResult objects
        """
        if self.structured_data is None:
            return []
            
        results = []
        
        # Prepare data frame for filtering
        df = self.structured_data.copy()
        
        # Apply filters if provided
        if filter_params:
            for col, val in filter_params.items():
                if col in df.columns:
                    df = df[df[col] == val]
        
        # Keywords in query for different operations
        query = query.lower()
        
        try:
            # Stats by game
            if "against" in query or "vs" in query:
                # Extract opponent name
                opponent = None
                if "against" in query:
                    opponent_part = query.split("against")[-1].strip().split()[0]
                    opponent = opponent_part.strip(".,?! ")
                elif "vs" in query:
                    opponent_part = query.split("vs")[-1].strip().split()[0]
                    opponent = opponent_part.strip(".,?! ")
                
                if opponent and "Opponent" in df.columns:
                    filtered_df = df[df["Opponent"].str.contains(opponent, case=False)]
                    if not filtered_df.empty:
                        for _, row in filtered_df.iterrows():
                            content = f"Game vs {row['Opponent']} ({row.get('Date', 'Unknown date')}):\n"
                            for col in row.index:
                                if col not in ['Game', 'Date', 'Opponent']:
                                    content += f"- {col}: {row[col]}\n"
                            
                            results.append(RetrievalResult(
                                content=content,
                                metadata={"source": "structured", "game": row.get('Game'), "opponent": row.get('Opponent')},
                                score=1.0
                            ))
            
            # Global stats
            elif any(term in query for term in ["average", "avg", "mean"]):
                # Get numeric columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                averages = df[numeric_cols].mean()
                
                content = "Team averages across all games:\n"
                for col, val in averages.items():
                    content += f"- Average {col}: {val:.2f}\n"
                
                results.append(RetrievalResult(
                    content=content,
                    metadata={"source": "structured", "operation": "average"},
                    score=1.0
                ))
                
            elif any(term in query for term in ["total", "sum"]):
                # Get numeric columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                sums = df[numeric_cols].sum()
                
                content = "Team totals across all games:\n"
                for col, val in sums.items():
                    content += f"- Total {col}: {val:.0f}\n"
                
                results.append(RetrievalResult(
                    content=content,
                    metadata={"source": "structured", "operation": "sum"},
                    score=1.0
                ))
                
            elif any(term in query for term in ["stats", "statistics", "numbers"]):
                # Get numeric columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                stats = df[numeric_cols].describe()
                
                content = "Team statistics summary:\n"
                for stat in ["mean", "min", "max"]:
                    if stat in stats.index:
                        content += f"\n{stat.capitalize()}:\n"
                        for col in numeric_cols:
                            content += f"- {col}: {stats.loc[stat, col]:.2f}\n"
                
                results.append(RetrievalResult(
                    content=content,
                    metadata={"source": "structured", "operation": "stats"},
                    score=1.0
                ))
            
            # Return all games as a fallback for structured data
            if not results:
                content = "All games:\n\n"
                for _, row in df.iterrows():
                    content += f"Game {row.get('Game')} vs {row.get('Opponent')} ({row.get('Date', 'Unknown date')}):\n"
                    for col in row.index:
                        if col not in ['Game', 'Date', 'Opponent']:
                            content += f"- {col}: {row[col]}\n"
                    content += "\n"
                
                results.append(RetrievalResult(
                    content=content,
                    metadata={"source": "structured", "operation": "all_games"},
                    score=0.5  # Lower score for fallback
                ))
            
            return results
            
        except Exception as e:
            logging.error(f"Error in structured data retrieval: {str(e)}")
            return []

    def _retrieve_from_unstructured(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks from unstructured data using vector similarity
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.unstructured_chunks or not self.faiss_index:
            return []
            
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype("float32")
            
            # Normalize the query vector
            faiss.normalize_L2(query_embedding)
            
            # Perform the search
            scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.unstructured_chunks)))
            
            # Collect results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.unstructured_chunks):  # Skip invalid indices
                    continue
                    
                content = self.unstructured_chunks[idx]
                metadata = self.unstructured_metadata[idx].copy()
                score = float(scores[0][i])
                
                results.append(RetrievalResult(
                    content=content,
                    metadata=metadata,
                    score=score
                ))
            
            return results
            
        except Exception as e:
            logging.error(f"Error in unstructured data retrieval: {str(e)}")
            return []

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Combined retrieval from structured and unstructured data
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects
        """
        # Check if query is better suited for structured data
        structured_keywords = ['statistics', 'stats', 'average', 'avg', 'total', 'percentage',
                             'rank', 'against', 'vs', 'numbers']
        is_structured_query = any(keyword in query.lower() for keyword in structured_keywords)
        
        results = []
        
        # Retrieve from both sources
        if is_structured_query and self.structured_data is not None:
            structured_results = self._retrieve_from_structured(query)
            results.extend(structured_results)
        
        # Always include unstructured search results as well
        unstructured_results = self._retrieve_from_unstructured(query, top_k)
        results.extend(unstructured_results)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        return results[:top_k]

    def answer_query(self, query: str, use_history: bool = True) -> Tuple[str, int]:
        """
        Answer a user query by retrieving relevant information and generating a response
        
        Args:
            query: User query
            use_history: Whether to include chat history for context
            
        Returns:
            Tuple of (answer, tokens_used)
        """
        try:
            # Log the query
            timestamp = datetime.now().isoformat()
            self.query_history.append({
                "timestamp": timestamp,
                "query": query
            })
            
            # Retrieve relevant information
            retrieval_results = self.retrieve(query)
            
            if not retrieval_results:
                answer = "I don't have enough information to answer that question."
                tokens_used = self._count_tokens(answer)
                return answer, tokens_used
            
            # Prepare context from retrieval results
            context = "\n\n".join([
                f"SOURCE [{i+1}] {result.metadata.get('source', 'unknown')}:\n{result.content}"
                for i, result in enumerate(retrieval_results)
            ])
            
            # Prepare chat history context if needed
            history_context = ""
            if use_history and self.chat_history:
                # Include up to last 3 exchanges to maintain context
                recent_history = self.chat_history[-3:]
                history_context = "\n".join([
                    f"User: {exchange['user']}\nAssistant: {exchange['assistant']}"
                    for exchange in recent_history
                ])
                history_context = "Previous conversation:\n" + history_context + "\n\n"
            
            # Prepare system prompt
            system_prompt = """
            You are an assistant providing information about Basketball.
            Answer questions based only on the provided context. 
            If the information isn't in the context, say you don't have that information.
            For each fact in your answer, indicate the source number in [brackets].
            Be concise, accurate, and helpful.
            """
            
            # Create the user prompt with context
            user_prompt = f"{history_context}Context information:\n{context}\n\nQuestion: {query}"
            
            # Generate the answer using OpenAI
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Calculate tokens used
            prompt_tokens = self._count_tokens(system_prompt) + self._count_tokens(user_prompt)
            completion_tokens = self._count_tokens(answer)
            total_tokens = prompt_tokens + completion_tokens
            
            # Update chat history
            self.chat_history.append({"user": query, "assistant": answer})
            
            return answer, total_tokens
            
        except Exception as e:
            logging.error(f"Error answering query: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}", 0

    def get_query_history(self) -> List[dict]:
        """Return the history of queries made to the system"""
        return self.query_history

    def get_chat_history(self) -> List[dict]:
        """Return the chat history"""
        return self.chat_history

    def save_query_history(self, file_path: str = "query_history.json") -> None:
        """Save the query history to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.query_history, f, indent=2)
            logging.info(f"Query history saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving query history: {str(e)}")
            raise

    def clear_chat_history(self) -> None:
        """Clear the chat history while preserving query history"""
        self.chat_history = []
        logging.info("Chat history cleared")