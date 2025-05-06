"""
Demo script for Enhanced Basketball RAG system with Llama

This script demonstrates the advanced functionality of the enhanced
basketball RAG system from the command line, using the Llama model
instead of OpenAI.
"""

import os
import argparse
from dotenv import load_dotenv
from basketball_rag import BasketballRAG
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Basketball RAG System Demo with Llama')
    parser.add_argument('--structured-data', type=str, default='data/structured/ucla_stats.csv',
                        help='Path to structured data file (CSV or Excel)')
    parser.add_argument('--unstructured-data', type=str, default='data/unstructured/game_summaries.txt',
                        help='Path to unstructured data file (text)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information about query processing')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name to use')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run models on (cpu or cuda)')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the RAG system
    print("\nInitializing Enhanced Basketball RAG System with Llama...")
    try:
        rag = BasketballRAG(
            llm_model=args.model,
            device=args.device
        )
    except Exception as e:
        logging.error(f"Error initializing RAG system: {str(e)}")
        print(f"\nFailed to initialize RAG system: {str(e)}")
        print("\nPlease make sure you have the necessary dependencies installed:")
        print("- transformers (pip install transformers)")
        print("- torch (pip install torch)")
        print("- sentence-transformers (pip install sentence-transformers)")
        print("- faiss-cpu or faiss-gpu (pip install faiss-cpu or pip install faiss-gpu)")
        return
    
    # Load data
    print(f"\nLoading structured data from {args.structured_data}...")
    try:
        rag.load_structured_data(args.structured_data)
    except Exception as e:
        logging.error(f"Error loading structured data: {str(e)}")
        print(f"\nFailed to load structured data: {str(e)}")
    
    print(f"\nLoading unstructured data from {args.unstructured_data}...")
    try:
        with open(args.unstructured_data, 'r', encoding='utf-8') as f:
            game_summaries = f.read()
        rag.load_unstructured_data(game_summaries)
    except Exception as e:
        logging.error(f"Error loading unstructured data: {str(e)}")
        print(f"\nFailed to load unstructured data: {str(e)}")
    
    # Display success message and data summary
    print("\nData loaded successfully!")
    
    if rag.structured_data is not None:
        print(f"Structured data: {len(rag.structured_data)} records with {len(rag.structured_data.columns)} fields")
    
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
        "Show me stats for games where players played more than 30 minutes",
        "How has scoring improved throughout the season?",
        "Which opponent did we have the highest field goal percentage against?"
    ]
    
    if args.interactive:
        # Interactive mode
        print("\n\nEnhanced Basketball RAG System - Interactive Mode")
        print("=" * 60)
        print("Type your questions or 'quit' to exit.")
        print("This version supports advanced analytics using the Llama model.")
        print("-" * 60)
        
        while True:
            query = input("\n> ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("\nProcessing query...")
            
            # Show verbose information if requested
            if args.verbose:
                # Analyze the query
                query_analysis = rag._analyze_query(query)
                print(f"\nQuery analysis: {query_analysis}")
            
            # Get the answer
            try:
                answer, tokens = rag.answer_query(query)
                
                print("\n" + "=" * 60)
                print(f"Answer: {answer}")
                print("-" * 60)
                print(f"Tokens used: {tokens}")
                print("=" * 60)
            except Exception as e:
                logging.error(f"Error processing query: {str(e)}")
                print(f"\nError: {str(e)}")
    else:
        # Demo mode with sample queries
        print("\n\nEnhanced Basketball RAG System Demo with Llama")
        print("=" * 60)
        
        total_tokens = 0
        for i, query in enumerate(sample_queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 60)
            
            # Show verbose information if requested
            if args.verbose:
                # Analyze the query
                query_analysis = rag._analyze_query(query)
                print(f"\nQuery analysis: {query_analysis}")
            
            try:
                answer, tokens = rag.answer_query(query)
                total_tokens += tokens
                
                print(f"Answer: {answer}")
                print(f"Tokens used: {tokens}")
                print("-" * 60)
            except Exception as e:
                logging.error(f"Error processing query: {str(e)}")
                print(f"Error: {str(e)}")
        
        print(f"\nTotal tokens used: {total_tokens}")
    
    # Save query history
    try:
        rag.save_query_history()
        print("\nQuery history has been saved to query_history.json")
    except Exception as e:
        logging.error(f"Error saving query history: {str(e)}")
        print(f"\nFailed to save query history: {str(e)}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()