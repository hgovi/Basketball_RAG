import os
import pandas as pd
from typing import Union, List
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basketball_rag.log'),
        logging.StreamHandler()
    ]
)

class BasketballRAG:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.structured_data = None
        self.unstructured_data = None
        self.vectorizer = TfidfVectorizer()
        self.text_embeddings = None
        self.query_history = []
        logging.info("BasketballRAG system initialized")
        
    def load_structured_data(self, file_path: str):
        """Load structured data from CSV or Excel file"""
        try:
            if file_path.endswith('.csv'):
                self.structured_data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.structured_data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
            logging.info(f"Successfully loaded structured data from {file_path}")
        except Exception as e:
            logging.error(f"Error loading structured data: {str(e)}")
            raise
            
    def load_unstructured_data(self, text_data: Union[str, List[str]]):
        """Load unstructured text data"""
        try:
            if isinstance(text_data, str):
                self.unstructured_data = [text_data]
            else:
                self.unstructured_data = text_data
                
            # Create embeddings for text data
            if self.unstructured_data:
                self.text_embeddings = self.vectorizer.fit_transform(self.unstructured_data)
                logging.info(f"Successfully loaded {len(self.unstructured_data)} unstructured documents")
        except Exception as e:
            logging.error(f"Error loading unstructured data: {str(e)}")
            raise
            
    def _process_structured_query(self, query: str) -> str:
        """Process queries that require structured data analysis"""
        if self.structured_data is None:
            return "No structured data available."
            
        try:
            # Enhanced query processing
            query = query.lower()
            
            # Get numeric columns for calculations
            numeric_cols = self.structured_data.select_dtypes(include=['int64', 'float64']).columns
            
            if "statistics" in query or "stats" in query:
                return self.structured_data[numeric_cols].describe().to_string()
            elif "average" in query:
                return self.structured_data[numeric_cols].mean().to_string()
            elif "total" in query:
                return self.structured_data[numeric_cols].sum().to_string()
            elif "against" in query:
                # Extract opponent name from query
                opponent = query.split("against")[-1].strip()
                return self.structured_data[self.structured_data['Opponent'].str.lower() == opponent].to_string()
            else:
                return "Please provide a more specific query about the structured data."
        except Exception as e:
            logging.error(f"Error processing structured query: {str(e)}")
            return f"Error processing structured query: {str(e)}"
            
    def _process_unstructured_query(self, query: str) -> str:
        """Process queries that require unstructured data analysis"""
        if self.unstructured_data is None:
            return "No unstructured data available."
            
        try:
            # Convert query to vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, self.text_embeddings)
            
            # Get most relevant document
            most_relevant_idx = np.argmax(similarity_scores)
            most_relevant_text = self.unstructured_data[most_relevant_idx]
            
            # Use OpenAI to generate a concise answer
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise answers about UCLA Women's Basketball."},
                    {"role": "user", "content": f"Based on this context: {most_relevant_text}\n\nAnswer this question: {query}"}
                ],
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error processing unstructured query: {str(e)}")
            return f"Error processing unstructured query: {str(e)}"
            
    def answer_query(self, query: str) -> tuple[str, int]:
        """Main method to answer user queries"""
        try:
            # Log the query
            self.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query
            })
            
            # Determine if query is better suited for structured or unstructured data
            structured_keywords = ['statistics', 'stats', 'average', 'total', 'percentage', 'rank', 'against']
            is_structured_query = any(keyword in query.lower() for keyword in structured_keywords)
            
            if is_structured_query and self.structured_data is not None:
                answer = self._process_structured_query(query)
                # Estimate tokens for structured query (roughly 1 token per 4 characters)
                tokens_used = len(answer) // 4
                return answer, tokens_used
            elif self.unstructured_data is not None:
                answer = self._process_unstructured_query(query)
                # Get actual token usage from OpenAI response
                tokens_used = len(answer) // 4  # Rough estimate
                return answer, tokens_used
            else:
                return "Insufficient data to answer accurately.", 0
        except Exception as e:
            logging.error(f"Error answering query: {str(e)}")
            return f"Error answering query: {str(e)}", 0
            
    def get_query_history(self) -> List[dict]:
        """Return the history of queries made to the system"""
        return self.query_history
        
    def save_query_history(self, file_path: str = "query_history.json"):
        """Save the query history to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.query_history, f, indent=2)
            logging.info(f"Query history saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving query history: {str(e)}")
            raise
