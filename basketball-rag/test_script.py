"""
Test script for the streamlined BasketballRAG system
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
    data_path = '../test/clean_oneseason.csv'  # Updated path to existing test data
    print(f"\nLoading structured data from {data_path}...")
    rag.load_structured_data(data_path)
    
    # Display data info
    print(f"\nLoaded {len(rag.structured_data)} games with {len(rag.structured_data.columns)} columns")
    print(f"Columns: {rag.structured_data.columns.tolist()}")
    
    # Optional: Load unstructured data if available
    try:
        text_path = 'data/game_summaries.txt'  # Update with your actual path
        if os.path.exists(text_path):
            print(f"\nLoading unstructured data from {text_path}...")
            with open(text_path, 'r', encoding='utf-8') as f:
                game_summaries = f.read()
            rag.load_unstructured_data(game_summaries)
    except:
        print("No unstructured data loaded")
    
    # Test basic queries
    test_queries = [
        "How many home games did OKC play?",
        "List the games against Houston this season",
        "When did OKC play against Dallas?",
        "Show me the statistics for games in March"
    ]
    
    print("\n=== Testing Basic Queries ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        # Process query
        answer, tokens = rag.answer_query(query)
        
        print(f"Answer: {answer}")
        print(f"Tokens used: {tokens}")
        print("-" * 50)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()