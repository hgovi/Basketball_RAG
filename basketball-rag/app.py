"""
Enhanced Basketball RAG Web Application

This Flask application provides a web interface for the Enhanced Basketball RAG system,
allowing users to interact with both structured and unstructured basketball data.

Features:
- Clean, responsive UI
- Real-time query answering
- History tracking
- Token usage monitoring
- Source citation
- Export capabilities
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from basketball_rag import BasketballRAG
import pandas as pd
import argparse

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

# Initialize the RAG system
rag = BasketballRAG(openai_api_key=api_key)

# Token usage tracking
total_tokens_used = 0

# Configure app with command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run the Basketball RAG Web Application')
    parser.add_argument('--structured-data', type=str, default='data/structured/ucla_stats.csv',
                        help='Path to structured data file (CSV or Excel)')
    parser.add_argument('--unstructured-data', type=str, default='data/unstructured/game_summaries.txt',
                        help='Path to unstructured data file (text)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to run the Flask application on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    return parser.parse_args()

def load_data(args):
    """Load data into the RAG system"""
    # Load structured data
    if os.path.exists(args.structured_data):
        rag.load_structured_data(args.structured_data)
    else:
        app.logger.warning(f"Structured data file not found: {args.structured_data}")
    
    # Load unstructured data
    if os.path.exists(args.unstructured_data):
        with open(args.unstructured_data, 'r', encoding='utf-8') as f:
            game_summaries = f.read()
        rag.load_unstructured_data(game_summaries)
    else:
        app.logger.warning(f"Unstructured data file not found: {args.unstructured_data}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    """Process a user query and return the answer"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Get conversation context if needed
        use_history = data.get('use_history', True)
        
        # Process the query through the RAG system
        answer, tokens_used = rag.answer_query(query, use_history=use_history)
        
        # Update token usage
        global total_tokens_used
        total_tokens_used += tokens_used
        
        return jsonify({
            'answer': answer,
            'tokens_used': tokens_used,
            'total_tokens_used': total_tokens_used
        })
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get the query history"""
    try:
        history = rag.get_query_history()
        return jsonify({'history': history})
    except Exception as e:
        app.logger.error(f"Error getting history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    """Get the full chat history with questions and answers"""
    try:
        chat_history = rag.get_chat_history()
        return jsonify({'chat_history': chat_history})
    except Exception as e:
        app.logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    """Clear the chat history"""
    try:
        rag.clear_chat_history()
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the data and system usage"""
    try:
        stats = {
            'games_analyzed': 0,
            'win_rate': 0,
            'avg_points': 0,
            'tokens_used': total_tokens_used
        }
        
        # Add stats from structured data if available
        if rag.structured_data is not None:
            df = rag.structured_data
            stats['games_analyzed'] = len(df)
            
            # Calculate win rate if there's a 'Win' column
            if 'Win' in df.columns:
                stats['win_rate'] = df['Win'].mean() * 100
            
            # Calculate average points if there's a 'Points' column
            if 'Points' in df.columns:
                stats['avg_points'] = df['Points'].mean()
        
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-history', methods=['GET'])
def download_history():
    """Save and return the query history as a JSON file"""
    try:
        rag.save_query_history('static/downloads/query_history.json')
        return send_from_directory('static/downloads', 'query_history.json', as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error downloading history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/img/<path:filename>')
def serve_images(filename):
    """Serve static image files"""
    return send_from_directory('static/img', filename)

@app.route('/static/downloads/<path:filename>')
def serve_downloads(filename):
    """Serve downloadable files"""
    return send_from_directory('static/downloads', filename)

def main():
    """Main function to run the Flask application"""
    # Parse command line arguments
    args = parse_args()
    
    # Ensure downloads directory exists
    os.makedirs('static/downloads', exist_ok=True)
    
    # Load data
    load_data(args)
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()