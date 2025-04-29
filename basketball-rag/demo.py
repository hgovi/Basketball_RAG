"""
Demo script for Enhanced Basketball RAG system

This script demonstrates the advanced functionality of the enhanced
basketball RAG system from the command line, with support for complex
analytical queries and statistical calculations.
"""

import os
import argparse
from dotenv import load_dotenv
from basketball_rag import BasketballRAG

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Basketball RAG System Demo')
    parser.add_argument('--structured-data', type=str, default='data/structured/ucla_stats.csv',
                        help='Path to structured data file (CSV or Excel)')
    parser.add_argument('--unstructured-data', type=str, default='data/unstructured/game_summaries.txt',
                        help='Path to unstructured data file (text)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information about query processing')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    # Initialize the RAG system
    print("\nInitializing Enhanced Basketball RAG System...")
    rag = BasketballRAG(openai_api_key=api_key)
    
    # Load data
    print(f"\nLoading structured data from {args.structured_data}...")
    rag.load_structured_data(args.structured_data)
    
    print(f"\nLoading unstructured data from {args.unstructured_data}...")
    with open(args.unstructured_data, 'r', encoding='utf-8') as f:
        game_summaries = f.read()
    rag.load_unstructured_data(game_summaries)
    
    # Display success message and data summary
    print("\nData loaded successfully!")
    
    if rag.structured_data is not None:
        print(f"Structured data: {len(rag.structured_data)} records with {len(rag.structured_data.columns)} fields")
        print(f"Derived features: {len(rag.derived_features)} additional metrics calculated")
    
    if rag.unstructured_chunks:
        print(f"Unstructured data: {len(rag.unstructured_chunks)} text chunks indexed")
    
    # Enhanced sample queries showcasing advanced capabilities
    sample_queries = [
        "What are the team's average points per game?",
        "Compare our performance against Boston versus against Lakers",
        "What's the trend in three-point shooting percentage over the last 5 games?",
        "Calculate the correlation between points scored and rebounds",
        "In games where we scored more than 100 points, what was our defensive efficiency?",
        "What happened in the game against Boston?",
        "Show me Gordon Hayward's stats in games where he played more than 30 minutes",
        "How has Shai Gilgeous-Alexander's scoring improved throughout the season?",
        "Which opponent did we have the highest field goal percentage against?"
    ]
    
    if args.interactive:
        # Interactive mode
        print("\n\nEnhanced Basketball RAG System - Interactive Mode")
        print("=" * 60)
        print("Type your questions or 'quit' to exit.")
        print("This version supports advanced analytics and statistical calculations.")
        print("-" * 60)
        
        while True:
            query = input("\n> ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("\nProcessing query...")
            
            # Show verbose information if requested
            if args.verbose:
                # Analyze the query
                query_analysis = rag._decompose_query(query)
                print(f"\nQuery analysis: {query_analysis}")
                
                # Show SQL query if applicable
                if query_analysis.get('query_type') in ['calculation', 'comparison', 'filtering']:
                    sql_query = rag._generate_sql_query(query_analysis)
                    if sql_query:
                        print(f"\nSQL query: {sql_query}")
            
            # Get the answer
            answer, tokens = rag.answer_query(query)
            
            print("\n" + "=" * 60)
            print(f"Answer: {answer}")
            print("-" * 60)
            print(f"Tokens used: {tokens}")
            print("=" * 60)
    else:
        # Demo mode with sample queries
        print("\n\nEnhanced Basketball RAG System Demo")
        print("=" * 60)
        
        total_tokens = 0
        for i, query in enumerate(sample_queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 60)
            
            # Show verbose information if requested
            if args.verbose:
                # Analyze the query
                query_analysis = rag._decompose_query(query)
                print(f"\nQuery analysis: {query_analysis}")
            
            answer, tokens = rag.answer_query(query)
            total_tokens += tokens
            
            print(f"Answer: {answer}")
            print(f"Tokens used: {tokens}")
            print("-" * 60)
        
        print(f"\nTotal tokens used: {total_tokens}")
    
    # Save query history
    rag.save_query_history()
    print("\nQuery history has been saved to query_history.json")
    print("\nDemo completed!")

if __name__ == "__main__":
    main()