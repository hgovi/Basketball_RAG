"""
Simple test script for the enhanced BasketballRAG system
"""

import os
from dotenv import load_dotenv
from basketball_rag import BasketballRAG

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

def main():
    # Initialize the RAG system
    print("\nInitializing Basketball RAG System...")
    rag = BasketballRAG(openai_api_key=api_key)
    
    # Load structured data
    data_path = './data/structured/ucla_stats.csv'  # This file actually contains OKC data
    print(f"\nLoading structured data from {data_path}...")
    rag.load_structured_data(data_path)
    
    # Display data info
    print(f"\nLoaded {len(rag.structured_data)} rows with {len(rag.structured_data.columns)} columns")
    print(f"Columns: {rag.structured_data.columns.tolist()}")
    
    # Test a few simple queries
    test_queries = [
        "How many home games did OKC play?",
        "List the games against Houston this season",
        "When did OKC play against DAL?",
        "What's the average number of days between games?"
    ]
    
    print("\n=== Testing Simple Queries ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        # Process query
        answer, tokens = rag.answer_query(query)
        
        print(f"Answer: {answer}")
        print(f"Tokens used: {tokens}")
        print("-" * 50)
    
    # Test specific components if needed
    print("\n=== Testing Query Decomposition ===")
    decomp_query = "Compare OKC's performance in home vs away games"
    query_analysis = rag._decompose_query(decomp_query)
    print(f"Query: {decomp_query}")
    print(f"Analysis: {query_analysis}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()