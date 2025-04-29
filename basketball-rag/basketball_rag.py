"""
Enhanced Basketball RAG System

This module combines the best elements of both RAG implementations with improvements:
1. Modern vector embeddings with FAISS indexing
2. Support for both structured and unstructured data
3. Intelligent query routing and decomposition
4. Advanced statistical analysis capabilities
5. Dynamic SQL generation for complex analytics
6. Enhanced context assembly with calculated results
7. Improved error handling and logging
8. Token usage tracking
9. Conversation history management
"""

import os
import pandas as pd
import numpy as np
import logging
import json
import sqlite3
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
from sqlalchemy import create_engine

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
        
        # SQL engine for advanced analytics
        self.sql_engine = None
        
        # Statistical functions registry
        self.stat_functions = {}
        
        # Derived features and statistics
        self.derived_features = {}
        
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
            
            # Preprocess the data
            self._preprocess_structured_data()
            
            # Initialize SQL engine for complex queries
            self._initialize_sql_engine()
            
            # Register statistical functions
            self._register_statistical_functions()
            
            logging.info(f"Successfully loaded structured data from {file_path} with {len(self.structured_data)} rows")
        except Exception as e:
            logging.error(f"Error loading structured data: {str(e)}")
            raise

    def _preprocess_structured_data(self) -> None:
        """Preprocess structured data for advanced analytics"""
        if self.structured_data is None:
            return
            
        try:
            # Make a copy of the original data
            self.original_structured_data = self.structured_data.copy()
            
            # 1. Handle missing values
            numeric_cols = self.structured_data.select_dtypes(include=['int64', 'float64']).columns
            self.structured_data[numeric_cols] = self.structured_data[numeric_cols].fillna(0)
            
            # 2. Generate derived features
            self.derived_features = {}
            
            # Shooting percentages
            if all(col in self.structured_data.columns for col in ['FGM', 'FGA']):
                self.structured_data['FG_PCT'] = self.structured_data['FGM'] / self.structured_data['FGA'].replace(0, 1)
                self.derived_features['FG_PCT'] = 'Field Goal Percentage (FGM/FGA)'
                
            if all(col in self.structured_data.columns for col in ['3PM', '3PA']):
                self.structured_data['3P_PCT'] = self.structured_data['3PM'] / self.structured_data['3PA'].replace(0, 1)
                self.derived_features['3P_PCT'] = '3-Point Percentage (3PM/3PA)'
                
            if all(col in self.structured_data.columns for col in ['FTM', 'FTA']):
                self.structured_data['FT_PCT'] = self.structured_data['FTM'] / self.structured_data['FTA'].replace(0, 1)
                self.derived_features['FT_PCT'] = 'Free Throw Percentage (FTM/FTA)'
            
            # Efficiency metrics
            if 'Points' in self.structured_data.columns and 'Minutes' in self.structured_data.columns:
                self.structured_data['PTS_PER_MIN'] = self.structured_data['Points'] / self.structured_data['Minutes'].replace(0, 1)
                self.derived_features['PTS_PER_MIN'] = 'Points Per Minute'
                
                self.structured_data['PTS_PER_36'] = self.structured_data['PTS_PER_MIN'] * 36
                self.derived_features['PTS_PER_36'] = 'Points Per 36 Minutes'
            
            # Advanced metrics
            if all(col in self.structured_data.columns for col in ['Points', 'FGA', 'FTA']):
                self.structured_data['TS_PCT'] = self.structured_data['Points'] / (2 * (self.structured_data['FGA'] + 0.44 * self.structured_data['FTA']))
                self.derived_features['TS_PCT'] = 'True Shooting Percentage'
            
            # Game-to-game differences
            if 'Game' in self.structured_data.columns:
                sorted_df = self.structured_data.sort_values('Game')
                for col in numeric_cols:
                    diff_col = f'{col}_DIFF'
                    self.structured_data[diff_col] = sorted_df[col].diff()
                    self.derived_features[diff_col] = f'Game-to-game difference in {col}'
            
            # 3. Compute rolling statistics if we have game sequence
            if 'Game' in self.structured_data.columns:
                sorted_df = self.structured_data.sort_values('Game')
                for col in numeric_cols:
                    # 3-game rolling average
                    roll_col = f'{col}_ROLL3'
                    self.structured_data[roll_col] = sorted_df[col].rolling(3, min_periods=1).mean()
                    self.derived_features[roll_col] = f'3-game rolling average of {col}'
            
            # 4. Calculate z-scores for normalized comparisons
            for col in numeric_cols:
                z_col = f'{col}_ZSCORE'
                self.structured_data[z_col] = (self.structured_data[col] - self.structured_data[col].mean()) / self.structured_data[col].std(ddof=0)
                self.derived_features[z_col] = f'Z-score of {col} (standard deviations from mean)'
            
            logging.info(f"Structured data preprocessing complete. Added {len(self.derived_features)} derived features.")
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            # Revert to original data if preprocessing fails
            if hasattr(self, 'original_structured_data'):
                self.structured_data = self.original_structured_data
            raise

    def _initialize_sql_engine(self) -> None:
        """Initialize SQL engine for complex analytics queries"""
        if self.structured_data is None:
            return
            
        try:
            # Create in-memory SQLite database
            self.sql_engine = create_engine('sqlite:///:memory:')
            
            # Store data in SQL
            self.structured_data.to_sql('basketball_stats', self.sql_engine, index=False, if_exists='replace')
            
            logging.info("SQL engine initialized with structured data")
        except Exception as e:
            logging.error(f"Error initializing SQL engine: {str(e)}")
            self.sql_engine = None

    def _register_statistical_functions(self) -> None:
        """Create a statistical analysis framework with advanced operations"""
        if self.structured_data is None:
            return
            
        # Create statistical functions registry
        self.stat_functions = {
            # Basic statistics
            'average': lambda df, col: df[col].mean(),
            'sum': lambda df, col: df[col].sum(),
            'min': lambda df, col: df[col].min(),
            'max': lambda df, col: df[col].max(),
            'median': lambda df, col: df[col].median(),
            'std': lambda df, col: df[col].std(),
            'var': lambda df, col: df[col].var(),
            
            # Advanced statistics
            'correlation': lambda df, x, y: df[x].corr(df[y]),
            'percentile': lambda df, col, p: df[col].quantile(p/100),
            'count_above': lambda df, col, threshold: (df[col] > threshold).sum(),
            'count_below': lambda df, col, threshold: (df[col] < threshold).sum(),
            'moving_average': lambda df, col, window: df.sort_values('Game')[col].rolling(window=window, min_periods=1).mean(),
            'pct_change': lambda df, col, periods: df.sort_values('Game')[col].pct_change(periods=periods),
            
            # Filtering operations
            'filter_equals': lambda df, col, value: df[df[col] == value],
            'filter_greater': lambda df, col, value: df[df[col] > value],
            'filter_less': lambda df, col, value: df[df[col] < value],
            'filter_between': lambda df, col, min_val, max_val: df[(df[col] >= min_val) & (df[col] <= max_val)],
            
            # Aggregation operations
            'group_avg': lambda df, group_col, agg_col: df.groupby(group_col)[agg_col].mean(),
            'group_sum': lambda df, group_col, agg_col: df.groupby(group_col)[agg_col].sum(),
            'group_min': lambda df, group_col, agg_col: df.groupby(group_col)[agg_col].min(),
            'group_max': lambda df, group_col, agg_col: df.groupby(group_col)[agg_col].max(),
            
            # Combined operations
            'win_loss_stats': lambda df, stat_col: {
                'win_avg': df[df['Win'] == 1][stat_col].mean(),
                'loss_avg': df[df['Win'] == 0][stat_col].mean(),
                'difference': df[df['Win'] == 1][stat_col].mean() - df[df['Win'] == 0][stat_col].mean()
            } if 'Win' in df.columns else None,
            
            'home_away_stats': lambda df, stat_col: {
                'home_avg': df[df['Home'] == 1][stat_col].mean(),
                'away_avg': df[df['Home'] == 0][stat_col].mean(),
                'difference': df[df['Home'] == 1][stat_col].mean() - df[df['Home'] == 0][stat_col].mean()
            } if 'Home' in df.columns else None
        }
        
        # Pre-compute common statistics for quick access
        self.structured_data_stats = {
            'summary': self.structured_data.describe(),
            'correlations': self.structured_data.select_dtypes(include=['int64', 'float64']).corr(),
        }
        
        # Create game-by-game summary if available
        if 'Game' in self.structured_data.columns:
            self.structured_data_stats['games'] = self.structured_data.set_index('Game').to_dict(orient='index')
        
        logging.info(f"Registered {len(self.stat_functions)} statistical functions")

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

    def _decompose_query(self, query: str) -> Dict:
        """
        Break down complex queries into components and identify required operations
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with query components and operation types
        """
        try:
            # Use LLM to decompose the query
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": """
                    You are a query analyzer for basketball analytics.
                    Analyze the query and break it down into components with the following structure:
                    {
                      "query_type": "calculation" | "retrieval" | "comparison" | "trend" | "filtering",
                      "operation": The specific operation needed (e.g., "average", "max", "correlation"),
                      "fields": ["field1", "field2"],
                      "filters": {"field": "value"},
                      "time_frame": "all" | "specific_game" | "recent_games",
                      "comparison_target": "opponent" | "player" | "time_period",
                      "structured_data_required": true/false,
                      "unstructured_data_required": true/false
                    }
                    Only include relevant fields and be precise.
                    """},
                    {"role": "user", "content": f"Analyze this basketball query: {query}"}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            
            # Extract JSON from the response if it's embedded in text
            import re
            json_match = re.search(r'({[\s\S]*})', analysis_text)
            
            if json_match:
                try:
                    analysis = json.loads(json_match.group(1))
                    logging.info(f"Query decomposed: {analysis}")
                    return analysis
                except json.JSONDecodeError:
                    logging.warning("Failed to parse JSON from LLM response")
            
            # Fallback to simple heuristic analysis if LLM fails
            return self._heuristic_query_analysis(query)
            
        except Exception as e:
            logging.error(f"Error in query decomposition: {str(e)}")
            return self._heuristic_query_analysis(query)

    def _heuristic_query_analysis(self, query: str) -> Dict:
        """
        Perform basic heuristic analysis of query when LLM decomposition fails
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with query type and components
        """
        query_lower = query.lower()
        
        # Check for calculation keywords
        calculation_keywords = ['average', 'avg', 'mean', 'total', 'sum', 'median', 'calculate', 
                               'how many', 'percentage', 'percent', 'ratio', 'difference']
        
        # Check for comparison keywords
        comparison_keywords = ['compare', 'versus', 'vs', 'against', 'difference between', 
                              'better', 'worse', 'higher', 'lower', 'more', 'less']
        
        # Check for trend keywords
        trend_keywords = ['trend', 'over time', 'change', 'progression', 'evolution', 
                         'improved', 'decreased', 'increased', 'pattern']
        
        # Check for filtering keywords
        filtering_keywords = ['only', 'filter', 'where', 'when', 'during', 'in games']
        
        # Determine query type
        if any(keyword in query_lower for keyword in calculation_keywords):
            query_type = "calculation"
        elif any(keyword in query_lower for keyword in comparison_keywords):
            query_type = "comparison"
        elif any(keyword in query_lower for keyword in trend_keywords):
            query_type = "trend"
        elif any(keyword in query_lower for keyword in filtering_keywords):
            query_type = "filtering"
        else:
            query_type = "retrieval"
        
        # Extract potential fields
        # This is a simplified approach - for production, use entity extraction
        fields = []
        for col in self.structured_data.columns if self.structured_data is not None else []:
            if col.lower() in query_lower:
                fields.append(col)
        
        # Create a simple analysis
        analysis = {
            "query_type": query_type,
            "fields": fields,
            "structured_data_required": query_type in ["calculation", "comparison", "trend"],
            "unstructured_data_required": query_type in ["retrieval"] or len(fields) == 0
        }
        
        logging.info(f"Heuristic query analysis: {analysis}")
        return analysis

    def _execute_statistical_operation(self, operation: str, query_components: Dict) -> str:
        """
        Execute a statistical operation based on decomposed query
        
        Args:
            operation: Name of the operation to execute
            query_components: Decomposed query with relevant fields
            
        Returns:
            A string with the result of the operation
        """
        if self.structured_data is None:
            return "No structured data available for calculations."
        
        try:
            df = self.structured_data
            result = None
            result_description = ""
            
            # Apply filters if specified
            if 'filters' in query_components and query_components['filters']:
                for field, value in query_components['filters'].items():
                    if field in df.columns:
                        df = df[df[field] == value]
            
            # Get fields to operate on
            fields = query_components.get('fields', [])
            if not fields and df is not None and not df.empty:
                # Default to numeric columns if none specified
                fields = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Basic operations
            if operation == 'average':
                result = {field: df[field].mean() for field in fields if field in df.columns}
                result_description = "Average values"
                
            elif operation == 'sum':
                result = {field: df[field].sum() for field in fields if field in df.columns}
                result_description = "Sum of values"
                
            elif operation == 'min':
                result = {field: df[field].min() for field in fields if field in df.columns}
                result_description = "Minimum values"
                
            elif operation == 'max':
                result = {field: df[field].max() for field in fields if field in df.columns}
                result_description = "Maximum values"
                
            elif operation == 'median':
                result = {field: df[field].median() for field in fields if field in df.columns}
                result_description = "Median values"
                
            # Advanced operations
            elif operation == 'correlation' and len(fields) >= 2:
                result = {f"{fields[0]}_{fields[1]}": df[fields[0]].corr(df[fields[1]])}
                result_description = f"Correlation between {fields[0]} and {fields[1]}"
                
            elif operation == 'trend' and 'Game' in df.columns:
                df = df.sort_values('Game')
                trend_results = {}
                for field in fields:
                    if field in df.columns:
                        # Calculate trend as a linear regression slope
                        x = df['Game'].values
                        y = df[field].values
                        slope = np.polyfit(x, y, 1)[0]
                        trend_results[field] = slope
                result = trend_results
                result_description = "Trend analysis (slope of linear regression)"
                
            elif operation == 'percentile' and 'percentile' in query_components:
                p = query_components['percentile']
                result = {field: df[field].quantile(p/100) for field in fields if field in df.columns}
                result_description = f"{p}th percentile"
                
            # Comparison operations
            elif operation == 'comparison' and 'comparison_target' in query_components:
                target = query_components['comparison_target']
                if target == 'opponent' and 'Opponent' in df.columns:
                    opponents = query_components.get('opponents', [])
                    if opponents:
                        comparison = {}
                        for opponent in opponents:
                            subset = df[df['Opponent'] == opponent]
                            comparison[opponent] = {field: subset[field].mean() for field in fields if field in subset}
                        result = comparison
                        result_description = "Comparison by opponent"
                
                elif target == 'time_period' and 'time_periods' in query_components:
                    # Implementation for time period comparison
                    pass
            
            # SQL-based operations
            elif operation == 'sql' and self.sql_engine is not None:
                sql_query = query_components.get('sql_query')
                if sql_query:
                    import pandas as pd
                    with self.sql_engine.connect() as conn:
                        result_df = pd.read_sql(sql_query, conn)
                    result = result_df.to_dict(orient='records')
                    result_description = "SQL query result"
            
            # Format the result for display
            if result is not None:
                formatted_result = f"{result_description}:\n"
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, dict):
                            formatted_result += f"\n{key}:\n"
                            for k, v in value.items():
                                formatted_result += f"  {k}: {v:.2f}\n"
                        else:
                            formatted_result += f"{key}: {value:.2f}\n"
                else:
                    formatted_result += str(result)
                
                return formatted_result
            else:
                return "Could not execute the requested statistical operation."
                
        except Exception as e:
            logging.error(f"Error executing statistical operation: {str(e)}")
            return f"Error in statistical calculation: {str(e)}"

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
        
        # Decompose the query to better understand intent
        query_analysis = self._decompose_query(query)
        query_type = query_analysis.get('query_type', 'retrieval')
        
        try:
            # Handle different query types
            if query_type == 'calculation':
                operation = query_analysis.get('operation', 'average')
                result = self._execute_statistical_operation(operation, query_analysis)
                
                results.append(RetrievalResult(
                    content=result,
                    metadata={"source": "structured_calculation", "operation": operation},
                    score=0.9
                ))
                
            elif query_type == 'comparison':
                # Process comparison queries
                comparison_target = query_analysis.get('comparison_target')
                fields = query_analysis.get('fields', [])
                
                if comparison_target == 'opponent' and 'Opponent' in df.columns:
                    # Find potential opponents in the query
                    all_opponents = df['Opponent'].unique().tolist()
                    mentioned_opponents = [opp for opp in all_opponents if opp.lower() in query.lower()]
                    
                    if not mentioned_opponents:
                        # Default to most recent opponents if none specified
                        mentioned_opponents = all_opponents[:2] if len(all_opponents) >= 2 else all_opponents
                    
                    # Compare metrics for these opponents
                    comparison_content = "Comparison by opponent:\n\n"
                    for opponent in mentioned_opponents:
                        opponent_df = df[df['Opponent'] == opponent]
                        comparison_content += f"Against {opponent}:\n"
                        
                        # Include relevant fields or all numeric

            elif query_type == 'comparison':
                # Process comparison queries
                comparison_target = query_analysis.get('comparison_target')
                fields = query_analysis.get('fields', [])
                
                if comparison_target == 'opponent' and 'Opponent' in df.columns:
                    # Find potential opponents in the query
                    all_opponents = df['Opponent'].unique().tolist()
                    mentioned_opponents = [opp for opp in all_opponents if opp.lower() in query.lower()]
                    
                    if not mentioned_opponents:
                        # Default to most recent opponents if none specified
                        mentioned_opponents = all_opponents[:2] if len(all_opponents) >= 2 else all_opponents
                    
                    # Compare metrics for these opponents
                    comparison_content = "Comparison by opponent:\n\n"
                    for opponent in mentioned_opponents:
                        opponent_df = df[df['Opponent'] == opponent]
                        comparison_content += f"Against {opponent}:\n"
                        
                        for field in fields or df.select_dtypes(include=['int64', 'float64']).columns[:5]:
                            if field in opponent_df.columns:
                                avg_value = opponent_df[field].mean()
                                comparison_content += f"- {field}: {avg_value:.2f}\n"
                        
                        comparison_content += "\n"
                    
                    results.append(RetrievalResult(
                        content=comparison_content,
                        metadata={"source": "structured_comparison", "target": "opponent"},
                        score=0.9
                    ))
                
                elif comparison_target == 'time_period' and 'Date' in df.columns:
                    # Implement time period comparisons
                    pass
            
            elif query_type == 'trend' and 'Game' in df.columns:
                # Process trend queries
                fields = query_analysis.get('fields', [])
                if not fields:
                    fields = df.select_dtypes(include=['int64', 'float64']).columns[:3].tolist()
                
                trend_content = "Trend analysis:\n\n"
                
                # Sort by game for trend analysis
                sorted_df = df.sort_values('Game')
                
                for field in fields:
                    if field in sorted_df.columns:
                        # Calculate simple linear regression
                        x = sorted_df['Game'].values
                        y = sorted_df[field].values
                        if len(x) > 1:  # Need at least two points
                            slope, intercept = np.polyfit(x, y, 1)
                            trend_direction = "increasing" if slope > 0 else "decreasing"
                            
                            trend_content += f"{field}: {trend_direction} trend (slope: {slope:.4f})\n"
                            trend_content += f"- First game: {y[0]:.2f}\n"
                            trend_content += f"- Last game: {y[-1]:.2f}\n"
                            trend_content += f"- Change: {y[-1] - y[0]:.2f}\n\n"
                
                results.append(RetrievalResult(
                    content=trend_content,
                    metadata={"source": "structured_trend", "fields": fields},
                    score=0.9
                ))
            
            elif query_type == 'filtering':
                # Process filtering queries
                filters = query_analysis.get('filters', {})
                fields = query_analysis.get('fields', [])
                
                filtered_df = df
                
                # Apply filters
                for field, value in filters.items():
                    if field in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df[field] == value]
                
                if len(filtered_df) > 0:
                    filter_content = f"Filtered results ({len(filtered_df)} entries):\n\n"
                    
                    # Show a summary of filtered data
                    for i, row in filtered_df.head(5).iterrows():
                        if 'Game' in row and 'Opponent' in row:
                            filter_content += f"Game {row['Game']} vs {row['Opponent']}:\n"
                        else:
                            filter_content += f"Entry {i}:\n"
                        
                        for field in fields or filtered_df.columns[:7]:
                            if field in row:
                                filter_content += f"- {field}: {row[field]}\n"
                        
                        filter_content += "\n"
                    
                    # Add summary statistics
                    filter_content += "Summary statistics for filtered data:\n"
                    numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
                    for col in numeric_cols[:5]:  # Limit to top 5 numeric columns
                        avg = filtered_df[col].mean()
                        filter_content += f"- Average {col}: {avg:.2f}\n"
                    
                    results.append(RetrievalResult(
                        content=filter_content,
                        metadata={"source": "structured_filtering", "filters": filters},
                        score=0.9
                    ))
                else:
                    results.append(RetrievalResult(
                        content="No data matches the specified filters.",
                        metadata={"source": "structured_filtering", "filters": filters},
                        score=0.5
                    ))
            
            # Fallback for structured data queries
            if not results:
                # Stats by game
                if "against" in query.lower() or "vs" in query.lower():
                    # Extract opponent name
                    opponent = None
                    if "against" in query.lower():
                        opponent_part = query.lower().split("against")[-1].strip().split()[0]
                        opponent = opponent_part.strip(".,?! ")
                    elif "vs" in query.lower():
                        opponent_part = query.lower().split("vs")[-1].strip().split()[0]
                        opponent = opponent_part.strip(".,?! ")
                    
                    if opponent and "Opponent" in df.columns:
                        # Find partial matches for opponent
                        filtered_df = df[df["Opponent"].str.lower().str.contains(opponent.lower())]
                        if not filtered_df.empty:
                            for _, row in filtered_df.iterrows():
                                content = f"Game vs {row['Opponent']} ({row.get('Date', 'Unknown date')}):\n"
                                for col in row.index:
                                    if col not in ['Game', 'Date', 'Opponent']:
                                        content += f"- {col}: {row[col]}\n"
                                
                                results.append(RetrievalResult(
                                    content=content,
                                    metadata={"source": "structured", "game": row.get('Game'), "opponent": row.get('Opponent')},
                                    score=0.8
                                ))
                
                # Global stats
                elif any(term in query.lower() for term in ["average", "avg", "mean"]):
                    # Get numeric columns
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    averages = df[numeric_cols].mean()
                    
                    content = "Team averages across all games:\n"
                    for col, val in averages.items():
                        content += f"- Average {col}: {val:.2f}\n"
                    
                    results.append(RetrievalResult(
                        content=content,
                        metadata={"source": "structured", "operation": "average"},
                        score=0.7
                    ))
                    
                elif any(term in query.lower() for term in ["total", "sum"]):
                    # Get numeric columns
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    sums = df[numeric_cols].sum()
                    
                    content = "Team totals across all games:\n"
                    for col, val in sums.items():
                        content += f"- Total {col}: {val:.0f}\n"
                    
                    results.append(RetrievalResult(
                        content=content,
                        metadata={"source": "structured", "operation": "sum"},
                        score=0.7
                    ))
                    
                elif any(term in query.lower() for term in ["stats", "statistics", "numbers"]):
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
                        score=0.7
                    ))
            
            # Return all games as a fallback for structured data
            if not results:
                content = "All games:\n\n"
                for _, row in df.head(5).iterrows():  # Limit to top 5 games
                    content += f"Game {row.get('Game')} vs {row.get('Opponent')} ({row.get('Date', 'Unknown date')}):\n"
                    for col in row.index[:7]:  # Limit to first 7 columns
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

    def _generate_sql_query(self, query_analysis: Dict) -> str:
        """
        Generate SQL query based on query analysis
        
        Args:
            query_analysis: Decomposed query information
            
        Returns:
            SQL query string
        """
        if self.sql_engine is None:
            return ""
            
        try:
            query_type = query_analysis.get('query_type', '')
            fields = query_analysis.get('fields', [])
            filters = query_analysis.get('filters', {})
            
            # Default to all columns if none specified
            if not fields and self.structured_data is not None:
                fields = self.structured_data.columns.tolist()[:5]  # Limit to first 5 for readability
            
            # Build the SQL query
            select_clause = "SELECT " + ", ".join(fields) if fields else "SELECT *"
            from_clause = "FROM basketball_stats"
            where_clause = ""
            
            # Add filters to WHERE clause
            if filters:
                conditions = []
                for field, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(f"{field} = '{value}'")
                    else:
                        conditions.append(f"{field} = {value}")
                
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)
            
            # Add aggregation for calculation queries
            if query_type == 'calculation':
                operation = query_analysis.get('operation', '')
                if operation == 'average':
                    select_clause = "SELECT " + ", ".join([f"AVG({field}) as avg_{field}" for field in fields])
                elif operation == 'sum':
                    select_clause = "SELECT " + ", ".join([f"SUM({field}) as sum_{field}" for field in fields])
                elif operation == 'min':
                    select_clause = "SELECT " + ", ".join([f"MIN({field}) as min_{field}" for field in fields])
                elif operation == 'max':
                    select_clause = "SELECT " + ", ".join([f"MAX({field}) as max_{field}" for field in fields])
            
            # Combine the clauses
            sql_query = f"{select_clause} {from_clause}"
            if where_clause:
                sql_query += f" {where_clause}"
            
            return sql_query
            
        except Exception as e:
            logging.error(f"Error generating SQL query: {str(e)}")
            return ""

    def _execute_sql_query(self, sql_query: str) -> str:
        """
        Execute SQL query and format the results
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Formatted query results
        """
        if self.sql_engine is None or not sql_query:
            return "SQL engine not available or query is empty."
        
        try:
            import pandas as pd
            with self.sql_engine.connect() as conn:
                result_df = pd.read_sql(sql_query, conn)
            
            if result_df.empty:
                return "No results found for the SQL query."
            
            # Format the results
            if len(result_df) == 1:
                # Single row result (e.g., from aggregation)
                result_str = "SQL Query Result:\n"
                for col in result_df.columns:
                    result_str += f"- {col}: {result_df.iloc[0][col]}\n"
            else:
                # Multiple row result
                result_str = f"SQL Query Results ({len(result_df)} rows):\n\n"
                
                # Show first 5 rows
                for i, row in result_df.head(5).iterrows():
                    result_str += f"Row {i+1}:\n"
                    for col in result_df.columns:
                        result_str += f"- {col}: {row[col]}\n"
                    result_str += "\n"
                
                if len(result_df) > 5:
                    result_str += f"... and {len(result_df) - 5} more rows\n"
            
            return result_str
            
        except Exception as e:
            logging.error(f"Error executing SQL query: {str(e)}")
            return f"Error executing SQL query: {str(e)}"

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Enhanced retrieval from structured and unstructured data
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects
        """
        # Analyze the query
        query_analysis = self._decompose_query(query)
        query_type = query_analysis.get('query_type', 'retrieval')
        structured_required = query_analysis.get('structured_data_required', False)
        unstructured_required = query_analysis.get('unstructured_data_required', True)
        
        results = []
        
        # For queries that need calculations or structured analysis
        if structured_required and self.structured_data is not None:
            # Try SQL approach for complex queries
            if query_type in ['calculation', 'filtering', 'comparison'] and self.sql_engine is not None:
                sql_query = self._generate_sql_query(query_analysis)
                if sql_query:
                    sql_result = self._execute_sql_query(sql_query)
                    results.append(RetrievalResult(
                        content=sql_result,
                        metadata={"source": "sql", "query": sql_query},
                        score=0.9
                    ))
            
            # Also use the standard structured retrieval
            structured_results = self._retrieve_from_structured(query)
            results.extend(structured_results)
        
        # Include unstructured data retrieval if needed or for general queries
        if unstructured_required or query_type == 'retrieval':
            unstructured_results = self._retrieve_from_unstructured(query, top_k)
            results.extend(unstructured_results)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        return results[:top_k]

    def _assemble_enhanced_context(self, retrieval_results: List[RetrievalResult], query: str) -> str:
        """
        Assemble an enhanced context for the LLM that combines structured and unstructured data
        
        Args:
            retrieval_results: List of retrieval results
            query: Original user query
            
        Returns:
            Enhanced context string
        """
        # Organize results by source type
        sources = {
            "structured_calculation": [],
            "structured_comparison": [],
            "structured_trend": [],
            "structured_filtering": [],
            "sql": [],
            "structured": [],
            "unstructured": []
        }
        
        # Group results by source type
        for i, result in enumerate(retrieval_results):
            source_type = result.metadata.get('source', 'unstructured')
            sources[source_type].append((i+1, result))
        
        # Assemble the context with clear section headings
        context = f"Query: {query}\n\n"
        
        # First add calculation results
        if sources['structured_calculation'] or sources['sql']:
            context += "CALCULATED STATISTICS:\n"
            for idx, result in sources['structured_calculation'] + sources['sql']:
                context += f"SOURCE [{idx}] {result.metadata.get('operation', 'calculation')}:\n{result.content}\n\n"
        
        # Add comparison results
        if sources['structured_comparison']:
            context += "COMPARISONS:\n"
            for idx, result in sources['structured_comparison']:
                context += f"SOURCE [{idx}] comparison:\n{result.content}\n\n"
        
        # Add trend results
        if sources['structured_trend']:
            context += "TRENDS:\n"
            for idx, result in sources['structured_trend']:
                context += f"SOURCE [{idx}] trend analysis:\n{result.content}\n\n"
        
        # Add filtering results
        if sources['structured_filtering']:
            context += "FILTERED DATA:\n"
            for idx, result in sources['structured_filtering']:
                context += f"SOURCE [{idx}] filtered results:\n{result.content}\n\n"
        
        # Add general structured data
        if sources['structured']:
            context += "STRUCTURED DATA:\n"
            for idx, result in sources['structured']:
                context += f"SOURCE [{idx}] {result.metadata.get('operation', 'data')}:\n{result.content}\n\n"
        
        # Add unstructured data
        if sources['unstructured']:
            context += "GAME SUMMARIES AND NARRATIVES:\n"
            for idx, result in sources['unstructured']:
                context += f"SOURCE [{idx}] text chunk {result.metadata.get('chunk_id', '')}:\n{result.content}\n\n"
        
        return context

    def answer_query(self, query: str, use_history: bool = True) -> Tuple[str, int]:
        """
        Enhanced query answering with improved context assembly and prompt engineering
        
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
            
            # Analyze the query
            query_analysis = self._decompose_query(query)
            query_type = query_analysis.get('query_type', 'retrieval')
            
            # Retrieve relevant information
            retrieval_results = self.retrieve(query)
            
            if not retrieval_results:
                answer = "I don't have enough information to answer that question."
                tokens_used = self._count_tokens(answer)
                return answer, tokens_used
            
            # Assemble enhanced context
            context = self._assemble_enhanced_context(retrieval_results, query)
            
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
            
            # Prepare tailored system prompt based on query type
            base_prompt = """
            You are an advanced basketball analytics assistant.
            Answer questions based on the provided context which includes both retrieved information and computed statistics.
            For each fact in your answer, indicate the source number in [brackets].
            Be concise, accurate, and helpful.
            """
            
            query_specific_instructions = ""
            if query_type == 'calculation':
                query_specific_instructions = """
                For this calculation query:
                - Explain the methodology and what was calculated
                - Present the numerical results clearly
                - If relevant, compare the results to typical values
                """
            elif query_type == 'comparison':
                query_specific_instructions = """
                For this comparison query:
                - Highlight key differences between the items being compared
                - Use relative terms (higher/lower, better/worse) when appropriate
                - Consider providing a conclusion about which is superior for relevant metrics
                """
            elif query_type == 'trend':
                query_specific_instructions = """
                For this trend analysis:
                - Describe the direction and magnitude of the trend
                - Note any significant patterns or anomalies
                - If possible, suggest causes or implications of the trend
                """
            
            system_prompt = base_prompt + query_specific_instructions
            
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