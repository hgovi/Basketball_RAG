"""
Demo script for Basketball RAG system

This script demonstrates the basic functionality of the enhanced
basketball RAG system from the command line.
"""

import os
import argparse
from dotenv import load_dotenv
from basketball_rag import BasketballRAG

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Basketball RAG System Demo')
    parser.add_argument('--structured-data', type=str, default='data/structured/ucla_stats.csv',
                        help='Path to structured data file (CSV or Excel)')
    parser.add_argument('--unstructured-data', type=str, default='data/unstructured/game_summaries.txt',
                        help='Path to unstructured data file (text)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    # Initialize the RAG system
    print("\nInitializing OKC Thunder Basketball RAG System...")
    rag = BasketballRAG(openai_api_key=api_key)
    
    # Load data
    print(f"\nLoading structured data from {args.structured_data}...")
    rag.load_structured_data(args.structured_data)
    
    print(f"\nLoading unstructured data from {args.unstructured_data}...")
    with open(args.unstructured_data, 'r', encoding='utf-8') as f:
        game_summaries = f.read()
    rag.load_unstructured_data(game_summaries)
    
    # Display success message
    print("\nData loaded successfully!")
    
    # Sample queries
    sample_queries = [
        "What are the team's average points per game?",
        "What happened in the game against Boston?",
        "Show me the statistics for Gordon Hayward",
        "What was Gordon Hayward's performance on April 3rd?",
        "How many assists did Shai Gilgeous-Alexander have in total?"
    ]
    
    if args.interactive:
        # Interactive mode
        print("\n\nOKC Thunder Basketball RAG System - Interactive Mode")
        print("=" * 50)
        print("Type your questions or 'quit' to exit.")
        print("-" * 50)
        
        while True:
            query = input("\n> ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("\nProcessing query...")
            answer, tokens = rag.answer_query(query)
            
            print("\n" + "=" * 50)
            print(f"Answer: {answer}")
            print("-" * 50)
            print(f"Tokens used: {tokens}")
            print("=" * 50)
    else:
        # Demo mode with sample queries
        print("\n\nOKC Thunder Basketball RAG System Demo")
        print("=" * 50)
        
        total_tokens = 0
        for i, query in enumerate(sample_queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 50)
            
            answer, tokens = rag.answer_query(query)
            total_tokens += tokens
            
            print(f"Answer: {answer}")
            print(f"Tokens used: {tokens}")
            print("-" * 50)
        
        print(f"\nTotal tokens used: {total_tokens}")
    
    # Save query history
    rag.save_query_history()
    print("\nQuery history has been saved to query_history.json")
    print("\nDemo completed!")

if __name__ == "__main__":
    main()