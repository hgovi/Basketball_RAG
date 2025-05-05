"""
Streamlined Basketball RAG System

A more efficient approach using embeddings for semantic understanding
and a flexible architecture for handling various query types.
"""

import os
import pandas as pd
import numpy as np
import logging
import json
from typing import Union, List, Tuple, Dict, Any, Optional
from datetime import datetime
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using numpy for vector search instead.")

# Try to import SentenceTransformer, but provide fallback
TRANSFORMER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    logging.warning("SentenceTransformer not available. Using simple text processing instead.")

import tiktoken
from openai import OpenAI
from dataclasses import dataclass

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
    """Streamlined RAG system for Basketball analytics"""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo"
    ):
        """Initialize the RAG system with models and data structures"""
        # API key setup
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Validate API key format
        if not self.openai_api_key.startswith(('sk-', 'org-')):
            logging.warning("API key format may be invalid. Check your API key.")
        
        # Initialize OpenAI client with retry settings
        self.client = OpenAI(
            api_key=self.openai_api_key,
            max_retries=5,  # Increase retries for rate limiting
            timeout=60.0    # Increase timeout
        )
        
        # Initialize data containers
        self.structured_data = None
        self.raw_data = None
        self.unstructured_chunks = []
        self.unstructured_metadata = []
        
        # Load embedding model if available
        if TRANSFORMER_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logging.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logging.error(f"Error loading embedding model: {str(e)}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            logging.warning("No embedding model available. Unstructured data search will be limited.")
        
        # Initialize FAISS index
        self.faiss_index = None
        
        # Set LLM model
        self.llm_model = llm_model
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(llm_model) if "gpt" in llm_model else None
        except:
            self.tokenizer = None
            logging.warning("Tokenizer not available. Token counting will be estimated.")
        
        # History tracking
        self.query_history = []
        self.chat_history = []
        
        # For compatibility with existing codebase
        self.derived_features = []
        self.sql_engine = None
        self.stat_functions = {
            'average': lambda df, col: df[col].mean(),
            'sum': lambda df, col: df[col].sum(),
            'min': lambda df, col: df[col].min(),
            'max': lambda df, col: df[col].max(),
            'median': lambda df, col: df[col].median(),
            'correlation': lambda df, col1, col2: df[col1].corr(df[col2]),
            'percentile': lambda df, col, p: df[col].quantile(p/100),
            'count_above': lambda df, col, val: (df[col] > val).sum(),
            'count_below': lambda df, col, val: (df[col] < val).sum(),
            'moving_average': lambda df, col, window: df[col].rolling(window=window).mean(),
            'pct_change': lambda df, col: df[col].pct_change(),
            'filter_equals': lambda df, col, val: df[df[col] == val],
            'filter_greater': lambda df, col, val: df[df[col] > val],
            'filter_less': lambda df, col, val: df[df[col] < val],
            'filter_between': lambda df, col, min_val, max_val: df[(df[col] >= min_val) & (df[col] <= max_val)],
            'group_avg': lambda df, group_col, val_col: df.groupby(group_col)[val_col].mean(),
            'group_sum': lambda df, group_col, val_col: df.groupby(group_col)[val_col].sum(),
        }
        
        logging.info(f"BasketballRAG initialized with model {llm_model}")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens for a given text using the appropriate tokenizer"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation
            return len(text) // 4

    def load_structured_data(self, file_path: str) -> None:
        """Load structured data from CSV or Excel file"""
        try:
            # Load data based on file extension
            if file_path.endswith('.csv'):
                self.raw_data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.raw_data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
            
            # Make a copy to work with
            self.structured_data = self.raw_data.copy()
            
            # Preprocess the data
            self._preprocess_data()
            
            logging.info(f"Loaded structured data from {file_path} with {len(self.structured_data)} rows")
        
        except Exception as e:
            logging.error(f"Error loading structured data: {str(e)}")
            raise

    def _preprocess_data(self) -> None:
        """Preprocess the structured data for analytics"""
        if self.structured_data is None:
            return
            
        try:
            # Handle missing values in numeric columns
            numeric_cols = self.structured_data.select_dtypes(include=['number']).columns
            self.structured_data[numeric_cols] = self.structured_data[numeric_cols].fillna(0)
            
            # Extract info from Matchup column if it exists
            if 'Matchup' in self.structured_data.columns:
                # Create Home/Away indicator
                self.structured_data['Home'] = self.structured_data['Matchup'].str.contains('vs').astype(int)
                
                # Extract opponent
                def extract_opponent(matchup):
                    if ' vs. ' in matchup:
                        return matchup.split(' vs. ')[1]
                    elif ' @ ' in matchup:
                        return matchup.split(' @ ')[1]
                    return None
                
                self.structured_data['Opponent'] = self.structured_data['Matchup'].apply(extract_opponent)
            
            # Convert date to datetime if present
            date_cols = [col for col in self.structured_data.columns if 'date' in col.lower()]
            if date_cols:
                for col in date_cols:
                    try:
                        self.structured_data[col] = pd.to_datetime(self.structured_data[col])
                    except:
                        pass
            
            logging.info("Data preprocessing completed")
        
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            # Revert to original data if preprocessing fails
            if self.raw_data is not None:
                self.structured_data = self.raw_data.copy()

    def load_unstructured_data(self, text_data: Union[str, List[str]], 
                              chunk_size: int = 1000, 
                              chunk_overlap: int = 200) -> None:
        """Load and chunk unstructured text data"""
        try:
            # Convert to list if it's a single string
            if isinstance(text_data, str):
                text_data = [text_data]
                
            # Clear existing data
            self.unstructured_chunks = []
            self.unstructured_metadata = []
            
            # Process each text document
            for doc_id, text in enumerate(text_data):
                chunks = self._create_chunks(text, chunk_size, chunk_overlap, doc_id)
                for chunk, metadata in chunks:
                    self.unstructured_chunks.append(chunk)
                    self.unstructured_metadata.append(metadata)
            
            # Build index for retrieval if embedding model is available
            if self.embedding_model is not None:
                self._build_index()
                logging.info(f"Loaded unstructured data: {len(self.unstructured_chunks)} chunks with embeddings")
            else:
                logging.info(f"Loaded unstructured data: {len(self.unstructured_chunks)} chunks without embeddings")
                logging.warning("No embedding model available. Unstructured data search will be limited.")
        
        except Exception as e:
            logging.error(f"Error loading unstructured data: {str(e)}")
            raise

    def _create_chunks(self, text: str, chunk_size: int, chunk_overlap: int, doc_id: int) -> List[Tuple[str, Dict]]:
        """Create chunks from text with overlap for better retrieval"""
        chunks = []
        
        if not text:
            return chunks
            
        if len(text) <= chunk_size:
            chunks.append((text, {"source": "unstructured", "doc_id": doc_id, "chunk_id": 0}))
            return chunks
        
        # Chunk the text
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            if end >= len(text):
                end = len(text)
            else:
                # Find a good breakpoint
                # Try to break at paragraph, then sentence, then word
                for separator in ["\n\n", ".", " "]:
                    last_sep = text.rfind(separator, start, end)
                    if last_sep != -1 and last_sep > start + (chunk_size // 3):
                        end = last_sep + 1  # Include the separator
                        break
            
            # Extract the chunk
            chunk = text[start:end].strip()
            
            # Add metadata
            metadata = {
                "source": "unstructured",
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "start_pos": start,
                "end_pos": end
            }
            
            chunks.append((chunk, metadata))
            chunk_id += 1
            
            # Move to next chunk with overlap
            start = end - chunk_overlap if end < len(text) else len(text)
        
        return chunks

    def _build_index(self) -> None:
        """Build a FAISS index for the chunks"""
        if not self.unstructured_chunks:
            return
            
        try:
            # Get embeddings
            if self.embedding_model:
                embeddings = self.embedding_model.encode(
                    self.unstructured_chunks, 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
            else:
                # Simple text processing if embedding model is not available
                embeddings = np.array([np.array([ord(c) for c in chunk]) for chunk in self.unstructured_chunks])
            
            # Create FAISS index
            if FAISS_AVAILABLE:
                embeddings = embeddings.astype(np.float32)
                faiss.normalize_L2(embeddings)
                
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(embeddings)
            else:
                # Use numpy for vector search if FAISS is not available
                self.faiss_index = embeddings
            
            logging.info(f"Built index with {len(self.unstructured_chunks)} chunks")
        
        except Exception as e:
            logging.error(f"Error building index: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Retrieve information from both structured and unstructured data"""
        results = []
        
        # Get structured data insights
        if self.structured_data is not None:
            struct_results = self._retrieve_structured_insights(query)
            results.extend(struct_results)
        
        # Get unstructured data insights
        if self.unstructured_chunks and self.faiss_index is not None:
            unstruct_results = self._retrieve_unstructured(query, top_k)
            results.extend(unstruct_results)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top results
        return results[:top_k]

    def _retrieve_structured_insights(self, query: str) -> List[RetrievalResult]:
        """Get relevant insights from structured data"""
        results = []
        
        if self.structured_data is None or self.structured_data.empty:
            return results
        
        try:
            # First try direct data analysis without API call
            insights = self._simple_data_analysis(query)
            
            # Only use API if simple analysis doesn't yield good results
            if insights.get('operation_type') == 'summary' and not insights.get('filters'):
                try:
                    # Use LLM to determine what kind of information might be useful
                    insights_prompt = f"""
                    I have structured basketball data with these columns: {', '.join(self.structured_data.columns)}
                    
                    For the query: "{query}"
                    
                    What insights or statistics would be most relevant? Provide your answer as a JSON object with these fields:
                    - relevant_columns: List of columns that are most relevant
                    - operation_type: One of "summary", "filtering", "comparison", "time_series", "specific_game"
                    - filters: Any filters to apply (column:value pairs)
                    - aggregations: Any aggregations to perform (mean, sum, etc.)
                    
                    Keep it concise and only include the JSON.
                    """
                    
                    response = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": "You are a data analysis assistant that returns only valid JSON."},
                            {"role": "user", "content": insights_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=250
                    )
                    
                    # Parse the response
                    import re
                    import json
                    
                    response_text = response.choices[0].message.content
                    # Extract JSON if it's embedded in explanatory text
                    json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
                    
                    if json_match:
                        try:
                            insights = json.loads(json_match.group(1))
                        except:
                            # Keep the simple analysis results
                            pass
                except Exception as api_error:
                    logging.warning(f"API call failed in structured insights: {str(api_error)}")
                    # Continue with simple analysis results
            
            # Apply the insights to get relevant data
            df = self.structured_data
            
            # Apply filters if any
            if 'filters' in insights and insights['filters']:
                for col, val in insights['filters'].items():
                    if col in df.columns:
                        if isinstance(val, list):
                            df = df[df[col].isin(val)]
                        else:
                            df = df[df[col] == val]
            
            # Get relevant columns
            columns = insights.get('relevant_columns', df.columns[:5].tolist())
            
            # Prepare the content based on operation type
            operation = insights.get('operation_type', 'summary')
            
            if operation == 'summary':
                # General summary statistics
                if all(col in df.columns for col in columns):
                    summary = df[columns].describe().to_string()
                    results.append(RetrievalResult(
                        content=f"Summary statistics for {', '.join(columns)}:\n{summary}",
                        metadata={"source": "structured", "operation": "summary"},
                        score=0.8
                    ))
            
            elif operation == 'filtering':
                # Filtered data
                if len(df) > 0:
                    filtered_data = df[columns].head(5).to_string()
                    results.append(RetrievalResult(
                        content=f"Filtered data ({len(df)} rows matched):\n{filtered_data}",
                        metadata={"source": "structured", "operation": "filtering"},
                        score=0.85
                    ))
            
            elif operation == 'comparison':
                # Comparisons (e.g., home vs away)
                if 'Home' in df.columns and len(df) > 0:
                    home_df = df[df['Home'] == 1]
                    away_df = df[df['Home'] == 0]
                    
                    comparison = "Home vs Away Comparison:\n"
                    for col in columns:
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            home_avg = home_df[col].mean()
                            away_avg = away_df[col].mean()
                            comparison += f"{col}: Home: {home_avg:.2f}, Away: {away_avg:.2f}, Diff: {home_avg-away_avg:.2f}\n"
                    
                    results.append(RetrievalResult(
                        content=comparison,
                        metadata={"source": "structured", "operation": "comparison"},
                        score=0.9
                    ))
            
            elif operation == 'specific_game':
                # Information about specific games
                if 'Game' in df.columns or 'Matchup' in df.columns:
                    col = 'Matchup' if 'Matchup' in df.columns else 'Game'
                    for _, row in df.head(3).iterrows():
                        game_info = f"Game information - {row[col]}:\n"
                        for column in columns:
                            if column in row:
                                game_info += f"{column}: {row[column]}\n"
                        
                        results.append(RetrievalResult(
                            content=game_info,
                            metadata={"source": "structured", "operation": "specific_game"},
                            score=0.85
                        ))
            
            # If no results yet, add a general summary as fallback
            if not results:
                df_summary = df.head(5).to_string()
                results.append(RetrievalResult(
                    content=f"Sample data (first 5 rows):\n{df_summary}",
                    metadata={"source": "structured", "operation": "fallback"},
                    score=0.6
                ))
            
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving structured insights: {str(e)}")
            # Fallback to simple approach
            return self._simple_structured_retrieval(query)

    def _simple_data_analysis(self, query: str) -> Dict:
        """Simple heuristic-based data analysis when LLM approach fails"""
        query = query.lower()
        
        # Default analysis
        analysis = {
            'relevant_columns': self.structured_data.columns[:5].tolist(),
            'operation_type': 'summary',
            'filters': {}
        }
        
        # Look for specific teams in the query
        if 'boston' in query or 'bos' in query:
            analysis['filters']['Opponent'] = 'BOS'
            analysis['operation_type'] = 'specific_game'
        elif 'lakers' in query or 'lal' in query:
            analysis['filters']['Opponent'] = 'LAL'
            analysis['operation_type'] = 'specific_game'
        
        # Look for comparison indicators
        if 'compare' in query or 'vs' in query or 'versus' in query:
            analysis['operation_type'] = 'comparison'
        
        # Look for specific game indicators
        if 'game against' in query or 'match against' in query:
            analysis['operation_type'] = 'specific_game'
        
        return analysis

    def _simple_structured_retrieval(self, query: str) -> List[RetrievalResult]:
        """Simple fallback retrieval from structured data"""
        results = []
        
        if self.structured_data is None or self.structured_data.empty:
            return results
        
        # Get a general summary
        columns_summary = ", ".join(self.structured_data.columns.tolist())
        
        results.append(RetrievalResult(
            content=f"Data contains {len(self.structured_data)} rows with columns: {columns_summary}",
            metadata={"source": "structured", "operation": "fallback_summary"},
            score=0.5
        ))
        
        # If query mentions specific teams, try to find those games
        query_lower = query.lower()
        if 'boston' in query_lower or 'bos' in query_lower:
            team = 'BOS'
        elif 'lakers' in query_lower or 'lal' in query_lower:
            team = 'LAL'
        else:
            team = None
            
        if team and 'Opponent' in self.structured_data.columns:
            team_games = self.structured_data[self.structured_data['Opponent'] == team]
            if not team_games.empty:
                team_summary = team_games.head(3).to_string()
                results.append(RetrievalResult(
                    content=f"Games against {team}:\n{team_summary}",
                    metadata={"source": "structured", "operation": "team_filter"},
                    score=0.7
                ))
        
        return results

    def _retrieve_unstructured(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Retrieve relevant chunks from unstructured data"""
        results = []
        
        if not self.unstructured_chunks or self.faiss_index is None:
            return results
            
        try:
            # Encode the query
            if self.embedding_model:
                query_embedding = self.embedding_model.encode([query])
                query_embedding = np.array(query_embedding).astype("float32")
            else:
                # Simple text processing if embedding model is not available
                query_embedding = np.array([np.array([ord(c) for c in query])])
            
            # Normalize embedding
            if FAISS_AVAILABLE:
                faiss.normalize_L2(query_embedding)
            else:
                # Normalize using numpy if FAISS is not available
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search the index
            if FAISS_AVAILABLE:
                scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.unstructured_chunks)))
                indices = indices[0]  # Get the first result row
                scores = scores[0]    # Get the first result row
            else:
                # Use numpy for vector search if FAISS is not available
                scores = np.dot(self.faiss_index, query_embedding.T).flatten()
                indices = np.argsort(-scores)[:top_k]
                scores = scores[indices]
            
            # Collect results
            for i, idx in enumerate(indices):
                if idx < 0 or idx >= len(self.unstructured_chunks):
                    continue
                
                content = self.unstructured_chunks[idx]
                metadata = self.unstructured_metadata[idx].copy()
                score = float(scores[i])
                
                results.append(RetrievalResult(
                    content=content,
                    metadata=metadata,
                    score=score
                ))
            
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving from unstructured data: {str(e)}")
            return []

    def answer_query(self, query: str, use_history: bool = True) -> Tuple[str, int]:
        """Answer a user query using retrieved information"""
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
                f"SOURCE [{i+1}] {result.metadata.get('source', 'unknown')} - {result.metadata.get('operation', 'data')}:\n{result.content}"
                for i, result in enumerate(retrieval_results)
            ])
            
            # Add chat history if needed
            history_context = ""
            if use_history and self.chat_history:
                recent_history = self.chat_history[-3:]
                history_context = "\n".join([
                    f"User: {exchange['user']}\nAssistant: {exchange['assistant']}"
                    for exchange in recent_history
                ])
                history_context = "Previous conversation:\n" + history_context + "\n\n"
            
            # Prepare system prompt
            system_prompt = """
            You are a basketball analytics assistant that provides insights based on the given data.
            Answer the question based on the provided context.
            For each fact in your answer, reference the source number in [brackets].
            If the information isn't in the context, say you don't have that information.
            Be concise, accurate, and helpful.
            """
            
            # Create user prompt with context
            user_prompt = f"{history_context}Context:\n{context}\n\nQuestion: {query}"
            
            try:
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
            except Exception as api_error:
                # Handle API errors (like rate limiting or quota exceeded)
                logging.error(f"OpenAI API error: {str(api_error)}")
                
                # Provide a fallback response based on the retrieved information
                answer = f"I encountered an API error, but here's what I found in my database:\n\n"
                for i, result in enumerate(retrieval_results[:3]):
                    answer += f"[{i+1}] {result.content[:200]}...\n\n"
                answer += "\nFor more detailed analysis, please try again later."
            
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

    def _decompose_query(self, query: str) -> Dict[str, Any]:
        """Minimal query analysis for compatibility with app.py"""
        query_lower = query.lower()
        
        # Default analysis
        analysis = {
            'query_type': 'factual',
            'required_fields': [],
            'filters': {},
            'time_period': None,
            'entities': []
        }
        
        # Simple heuristic detection without API call
        if any(term in query_lower for term in ['average', 'mean', 'sum', 'total']):
            analysis['query_type'] = 'calculation'
            
            # Try to identify relevant fields
            stat_terms = {
                'points': ['points', 'score', 'scoring'],
                'assists': ['assists', 'passing', 'pass'],
                'rebounds': ['rebounds', 'rebound', 'boards'],
                'steals': ['steals', 'steal'],
                'blocks': ['blocks', 'block'],
                'fieldGoalsPercentage': ['shooting', 'field goal', 'fg'],
                'threePointersPercentage': ['three', '3-point', '3pt']
            }
            
            for field, terms in stat_terms.items():
                if any(term in query_lower for term in terms):
                    analysis['required_fields'].append(field)
            
        elif any(term in query_lower for term in ['compare', 'vs', 'versus']):
            analysis['query_type'] = 'comparison'
            
            # Try to identify comparison entities
            if 'home' in query_lower and 'away' in query_lower:
                analysis['entities'] = ['Home', 'Away']
                
        elif any(term in query_lower for term in ['filter', 'where', 'when']):
            analysis['query_type'] = 'filtering'
            
            # Try to identify time periods
            time_periods = ['january', 'february', 'march', 'april', 'may', 'june',
                           'july', 'august', 'september', 'october', 'november', 'december']
            for period in time_periods:
                if period in query_lower:
                    analysis['time_period'] = period.capitalize()
        
        # Extract entities (teams)
        teams = ['boston', 'celtics', 'lakers', 'warriors', 'bulls', 'heat', 'knicks', 'okc', 'thunder']
        for team in teams:
            if team in query_lower:
                analysis['entities'].append(team.capitalize())
                
                # Add filter for team
                if 'Opponent' in analysis.get('filters', {}):
                    analysis['filters']['Opponent'] = team.upper()
        
        return analysis
    
    def _generate_sql_query(self, query_analysis: Dict[str, Any]) -> str:
        """Minimal SQL generation for compatibility with app.py"""
        return ""