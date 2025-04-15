from basketball_rag import BasketballRAG
import os

def main():
    # Initialize the RAG system with your OpenAI API key
    api_key = "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"
    rag = BasketballRAG(openai_api_key=api_key)
    
    # Load sample data
    structured_data_path = "data/structured/ucla_stats.csv"
    unstructured_data_path = "data/unstructured/game_summaries.txt"
    
    # Load structured data
    rag.load_structured_data(structured_data_path)
    
    # Load unstructured data
    with open(unstructured_data_path, 'r') as f:
        game_summaries = f.read()
    rag.load_unstructured_data(game_summaries)
    
    # Sample queries
    queries = [
        "What are the team's average points per game?",
        "What happened in the game against USC?",
        "Show me the statistics for the game against Stanford",
        "What was the team's performance in the last game?",
        "How many assists did the team have in total?"
    ]
    
    # Process and display answers
    print("\nUCLA Women's Basketball RAG System Demo")
    print("=" * 50)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        answer = rag.answer_query(query)
        print(f"Answer: {answer}")
    
    # Save query history
    rag.save_query_history()
    print("\nQuery history has been saved to query_history.json")

if __name__ == "__main__":
    main() 