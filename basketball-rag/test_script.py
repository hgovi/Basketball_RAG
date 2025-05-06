"""
Simple test script for the Basketball RAG system with Llama
"""

import os
import sys
from dotenv import load_dotenv
from basketball_rag import BasketballRAG

# Load environment variables
load_dotenv()

def main():
    # Initialize the RAG system
    print("\nInitializing Basketball RAG System with open-access model...")
    try:
        # Use an open-access model that doesn't require authentication
        rag = BasketballRAG(
            llm_model="mistralai/Mistral-7B-Instruct-v0.2",  # Better open-access model
            device="cpu"
        )
        print("BasketballRAG initialized successfully")
    except Exception as e:
        print(f"Error initializing BasketballRAG: {str(e)}")
        sys.exit(1)
    
    # Load structured data
    data_path = '../test/clean_oneseason.csv'
    print(f"\nLoading structured data from {data_path}...")
    try:
        rag.load_structured_data(data_path)
        print(f"Loaded {len(rag.structured_data)} games with {len(rag.structured_data.columns)} columns")
    except Exception as e:
        print(f"Error loading structured data: {str(e)}")
        sys.exit(1)
    
    # Test basic queries
    test_queries = [
        "How many home games did OKC play?",
        "List the games against Houston this season"
    ]
    
    print("\n=== Testing Basic Queries ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        # Process query
        try:
            answer, tokens = rag.answer_query(query)
            print(f"Answer: {answer}")
            print(f"Tokens used: {tokens}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
        print("-" * 50)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()