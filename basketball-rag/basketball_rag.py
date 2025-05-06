"""
Streamlined Basketball RAG System using Llama

A more efficient approach using Llama for LLM processing and embedding models
for semantic understanding, with simplified architecture for handling various query types.
"""

import os
import pandas as pd
import numpy as np
import logging
import json
from typing import Union, List, Tuple, Dict, Any, Optional
from datetime import datetime
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

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    logging.warning("SentenceTransformer not available. Install with 'pip install sentence-transformers'")

# Try to import FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")

# Hugging Face transformers for Llama model
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Hugging Face transformers not available. Install with 'pip install transformers torch'")


@dataclass
class RetrievalResult:
    """Structure for storing retrieval results"""
    content: str
    metadata: Dict[str, Any]
    score: float


class BasketballRAG:
    """Streamlined RAG system for Basketball analytics using Llama"""
    
    def __init__(
            self, 
            llm_model: str = "meta-llama/Llama-2-7b-chat-hf",
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            device: str = "cpu",
            context_window: int = 4096,
            max_new_tokens: int = 512
        ):
        """Initialize the RAG system with models and data structures
        
        Args:
            llm_model: Name of the Hugging Face model to use
            embedding_model: Name of the embedding model for semantic search
            device: Device to run models on ("cpu" or "cuda")
            context_window: Maximum context window size for the LLM
            max_new_tokens: Maximum number of tokens to generate in responses
        """
        self.device = device
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.llm_model = llm_model  # Store the model name
        
        # Initialize data containers
        self.structured_data = None
        self.raw_data = None
        self.unstructured_chunks = []
        self.unstructured_metadata = []
        
        # Load embedding model
        if TRANSFORMER_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.embedding_model.to(device)
                logging.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logging.error(f"Error loading embedding model: {str(e)}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            logging.warning("No embedding model available. Unstructured data search will be limited.")
        
        # Load LLM
        if HF_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
                if device == "cpu" or not torch.cuda.is_available():
                    # Use standard loading without quantization for CPU
                    self.model = AutoModelForCausalLM.from_pretrained(
                        llm_model, 
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map=device,
                        # Removed quantization for compatibility
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        llm_model,
                        torch_dtype=torch.float16,
                        device_map=device
                    )
                
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                logging.info(f"Loaded LLM: {llm_model}")
            except Exception as e:
                logging.error(f"Error loading LLM: {str(e)}")
                raise RuntimeError(f"Failed to load LLM model: {str(e)}")
        else:
            raise ImportError("Hugging Face transformers is required for this RAG system.")
        
        # Initialize FAISS index
        self.faiss_index = None
        
        # History tracking
        self.query_history = []
        self.chat_history = []
        
        # Statistical functions
        self.stat_functions = {
            'average': lambda df, col: df[col].mean(),
            'sum': lambda df, col: df[col].sum(),
            'min': lambda df, col: df[col].min(),
            'max': lambda df, col: df[col].max(),
            'median': lambda df, col: df[col].median(),
            'correlation': lambda df, col1, col2: df[col1].corr(df[col2]),
            'count_above': lambda df, col, val: (df[col] > val).sum(),
            'count_below': lambda df, col, val: (df[col] < val).sum(),
            'filter_equals': lambda df, col, val: df[df[col] == val],
            'filter_greater': lambda df, col, val: df[df[col] > val],
            'filter_less': lambda df, col, val: df[df[col] < val],
            'filter_between': lambda df, col, min_val, max_val: df[(df[col] >= min_val) & (df[col] <= max_val)],
            'group_avg': lambda df, group_col, val_col: df.groupby(group_col)[val_col].mean(),
            'group_sum': lambda df, group_col, val_col: df.groupby(group_col)[val_col].sum(),
        }
        
        logging.info("BasketballRAG initialized successfully")

    def count_tokens(self, text: str) -> int:
        """Count tokens for a given text using the LLM tokenizer"""
        return len(self.tokenizer.encode(text))

    def generate_text(self, prompt: str) -> str:
        """Generate text using the LLM"""
        # Check if prompt is too long
        if self.count_tokens(prompt) > self.context_window:
            logging.warning(f"Prompt exceeds context window ({self.count_tokens(prompt)} > {self.context_window})")
            # Truncate prompt to fit context window
            prompt = self.tokenizer.decode(
                self.tokenizer.encode(prompt)[:self.context_window - self.max_new_tokens - 10]
            )
        
        try:
            # Format prompt based on model type
            model_name = self.llm_model.lower()
            
            if "llama-2" in model_name:
                formatted_prompt = f"""<s>[INST] {prompt} [/INST]"""
            elif "tinyllama" in model_name:
                formatted_prompt = f"""<|system|>
You are a basketball analytics assistant that provides insights based on given data.
{prompt}"""
            elif "mistral" in model_name:
                formatted_prompt = f"""<s>[INST] {prompt} [/INST]"""
            else:
                # Default format for other models
                formatted_prompt = prompt
            
            # Generate text
            try:
                response = self.llm_pipeline(
                    formatted_prompt,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=self.max_new_tokens,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract text from response
                output_text = response[0]['generated_text']
                
                # Remove the prompt from the output
                output_text = output_text.replace(formatted_prompt, "").strip()
                
                # For TinyLlama, extract the user response part
                if "tinyllama" in model_name:
                    # Look for the <|assistant|> tag
                    if "<|assistant|>" in output_text:
                        output_text = output_text.split("<|assistant|>")[1].strip()
                        # Remove any trailing tags
                        if "<|" in output_text:
                            output_text = output_text.split("<|")[0].strip()
                
                return output_text
            except IndexError:
                # Handle index errors which can happen with some models
                return "Based on the data provided, I cannot give a specific answer to this question."
        
        except Exception as e:
            logging.error(f"Error generating text: {str(e)}")
            return f"Error generating response: {str(e)}"

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
                # Find a good breakpoint (paragraph, sentence, or word)
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
                return
            
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
        if self.unstructured_chunks and self.embedding_model is not None:
            unstruct_results = self._retrieve_unstructured(query, top_k)
            results.extend(unstruct_results)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top results
        return results[:top_k]

    def _retrieve_structured_insights(self, query: str) -> List[RetrievalResult]:
        """Get relevant insights from structured data using simple analysis"""
        results = []
        
        if self.structured_data is None or self.structured_data.empty:
            return results
        
        try:
            # Use a simple data analysis approach based on query keywords
            insights = self._analyze_query(query)
            
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
            
            # Select relevant columns
            columns = insights.get('relevant_columns', df.columns.tolist()[:5])
            
            # Generate content based on query type
            operation = insights.get('query_type', 'summary')
            
            if operation == 'summary':
                # General summary statistics
                if len(df) > 0:
                    numeric_cols = df[columns].select_dtypes(include=['number']).columns
                    if not numeric_cols.empty:
                        summary = df[numeric_cols].agg(['mean', 'min', 'max']).round(2)
                        summary_str = summary.to_string()
                        results.append(RetrievalResult(
                            content=f"Summary statistics for numeric columns:\n{summary_str}",
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
            return []

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine the type of information needed"""
        query_lower = query.lower()
        
        # Initialize analysis with defaults
        analysis = {
            'query_type': 'summary',
            'relevant_columns': [],
            'filters': {},
            'entities': []
        }
        
        # Determine query type based on keywords
        if any(word in query_lower for word in ['average', 'mean', 'avg', 'calculate', 'what is the']):
            analysis['query_type'] = 'calculation'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'better']):
            analysis['query_type'] = 'comparison'
        elif any(word in query_lower for word in ['list', 'show', 'find', 'filter', 'when', 'which games']):
            analysis['query_type'] = 'filtering'
        elif any(word in query_lower for word in ['what happened', 'tell me about', 'details']):
            analysis['query_type'] = 'specific_game'
        
        # Identify relevant columns
        stat_terms = {
            'Points': ['points', 'score', 'scoring', 'pts'],
            'Assists': ['assists', 'passing', 'pass', 'ast'],
            'Rebounds': ['rebounds', 'rebound', 'boards', 'reb'],
            'Steals': ['steals', 'steal', 'stl'],
            'Blocks': ['blocks', 'block', 'blk'],
            'FG_PCT': ['shooting', 'field goal', 'fg', 'shooting percentage'],
            'FG3_PCT': ['three', '3-point', '3pt', 'three point', '3-pointer']
        }
        
        # Add relevant columns based on query terms
        for col, terms in stat_terms.items():
            if any(term in query_lower for term in terms):
                analysis['relevant_columns'].append(col)
        
        # If no specific columns identified, use basic ones
        if not analysis['relevant_columns']:
            basic_cols = ['Game', 'Matchup', 'Points', 'Rebounds', 'Assists']
            analysis['relevant_columns'] = [col for col in basic_cols if col in self.structured_data.columns]
        
        # Look for team names
        teams = {
            'BOS': ['boston', 'celtics'],
            'LAL': ['lakers', 'la lakers', 'los angeles lakers'],
            'GSW': ['warriors', 'golden state'],
            'CHI': ['bulls', 'chicago'],
            'MIA': ['heat', 'miami'],
            'NYK': ['knicks', 'new york'],
            'OKC': ['thunder', 'okc', 'oklahoma']
        }
        
        for team_code, team_names in teams.items():
            if any(team in query_lower for team in team_names):
                analysis['filters']['Opponent'] = team_code
                analysis['entities'].append(team_code)
        
        # Look for player names if they exist in data
        if 'Player' in self.structured_data.columns:
            player_list = self.structured_data['Player'].unique()
            for player in player_list:
                if player.lower() in query_lower:
                    analysis['filters']['Player'] = player
                    analysis['entities'].append(player)
        
        return analysis

    def _retrieve_unstructured(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Retrieve relevant chunks from unstructured data"""
        results = []
        
        if not self.unstructured_chunks or self.embedding_model is None:
            return results
        
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype("float32")
            
            # Search the index
            if FAISS_AVAILABLE and self.faiss_index is not None:
                # Normalize the query embedding
                faiss.normalize_L2(query_embedding)
                
                # Perform the search
                scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.unstructured_chunks)))
                indices = indices[0]  # Get the first result row
                scores = scores[0]    # Get the first result row
            else:
                # Use numpy for vector search
                embeddings = self.faiss_index
                scores = np.dot(embeddings, query_embedding.T).flatten()
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
                tokens_used = self.count_tokens(answer)
                return answer, tokens_used
            
            # Prepare context from retrieval results
            context = "\n\n".join([
                f"SOURCE [{i+1}] {result.metadata.get('source', 'unknown')} - {result.metadata.get('operation', 'data')}:\n{result.content}"
                for i, result in enumerate(retrieval_results)
            ])
            
            # Add chat history if needed
            history_context = ""
            if use_history and self.chat_history:
                recent_history = self.chat_history[-3:]  # Last 3 exchanges
                history_context = "\n".join([
                    f"User: {exchange['user']}\nAssistant: {exchange['assistant']}"
                    for exchange in recent_history
                ])
                history_context = "Previous conversation:\n" + history_context + "\n\n"
            
            # Prepare prompt for the LLM
            prompt = f"""You are a basketball analytics assistant that provides insights based on given data.
            
{history_context}

Here is the information I have about basketball data:

{context}

Question: {query}

Please answer the question based on the provided information. For each fact in your answer, reference the source number in [brackets]. If the information isn't in the context, say you don't have that information. Be concise, accurate, and helpful.
"""
            
            # Generate the answer
            answer = self.generate_text(prompt)
            
            # Calculate tokens used (prompt + response)
            tokens_used = self.count_tokens(prompt) + self.count_tokens(answer)
            
            # Update chat history
            self.chat_history.append({"user": query, "assistant": answer})
            
            return answer, tokens_used
            
        except Exception as e:
            logging.error(f"Error answering query: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}", 0

    def get_query_history(self) -> List[dict]:
        """Return the history of queries made to the system"""
        return self.query_history

    def get_chat_history(self) -> List[dict]:
        """Return the chat history"""
        return self.chat_history

    def clear_chat_history(self) -> None:
        """Clear the chat history while preserving query history"""
        self.chat_history = []
        logging.info("Chat history cleared")

    def save_query_history(self, file_path: str = "query_history.json") -> None:
        """Save the query history to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.query_history, f, indent=2)
            logging.info(f"Query history saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving query history: {str(e)}")
            raise