#!/usr/bin/env python3
"""
UCLA Women's Basketball RAG Analytics Web Application

A professional RAG-powered chatbot for UCLA women's basketball statistics.
Uses advanced Natural Language Processing to answer complex basketball queries.

Author: Om
Created: 2024
Version: 1.0.0
"""

import os
import sys
import sqlite3
import logging
import traceback
from datetime import datetime
from threading import local
from flask import Flask, render_template, request, jsonify, session

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.db_connector import DatabaseConnector
    from src.rag_pipeline import RAGPipeline
    from src.llm_utils import LLMManager
    print("✅ Successfully imported RAG components")
except ImportError as e:
    print(f"❌ Error importing RAG components: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ucla_webapp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ucla-womens-basketball-rag-secret-key-2024')

# Thread-local storage for database connections (thread-safe)
thread_local = local()


def get_thread_safe_connection():
    """
    Get a thread-safe database connection.
    
    Returns:
        sqlite3.Connection: Thread-safe database connection
    """
    if not hasattr(thread_local, 'connection'):
        thread_local.connection = sqlite3.connect(
            'data/ucla_wbb.db', 
            check_same_thread=False
        )
        thread_local.connection.row_factory = sqlite3.Row
    return thread_local.connection


def init_session():
    """Initialize session variables for tracking user interactions."""
    if 'token_count' not in session:
        session['token_count'] = 0
    if 'chat_history' not in session:
        session['chat_history'] = []


@app.before_request
def before_request():
    """Initialize session before each request."""
    init_session()


@app.route('/')
def index():
    """
    Main application page.
    
    Returns:
        str: Rendered HTML template
    """
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    """
    Handle user queries using the RAG pipeline.
    
    Returns:
        json: Query response with generated answer and metadata
    """
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Please enter a question'}), 400
        
        logger.info(f"Processing query: {user_query}")
        
        # Process using the RAG pipeline
        response_data = process_with_rag_pipeline(user_query)
        
        # Count tokens and update session
        token_count = len(response_data.get('response', '').split())
        session['token_count'] += token_count
        
        # Add to chat history
        session['chat_history'].append({
            'timestamp': datetime.now().isoformat(),
            'query': user_query,
            'response': response_data.get('response', ''),
            'tokens': token_count
        })
        
        return jsonify({
            'response': response_data.get('response', 'Sorry, no response generated.'),
            'tokens': token_count,
            'total_tokens': session['token_count']
        })
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'response': 'Sorry, there was an error processing your question. Please try again.'
        }), 200


def process_with_rag_pipeline(user_query):
    """
    Process query using the RAG pipeline with intelligent entity extraction,
    SQL generation, and natural language response generation.
    
    Args:
        user_query (str): Natural language query from the user
        
    Returns:
        dict: Response data containing generated answer and success status
    """
    try:
        # Create fresh components for this request (thread-safe)
        db_connector = DatabaseConnector(db_path='data/ucla_wbb.db')
        db_connector.connect()
        
        try:
            llm_manager = LLMManager()
        except ValueError as e:
            # Handle LLM initialization error
            error_msg = str(e)
            if "ANTHROPIC_API_KEY" in error_msg:
                return {
                    'response': (
                        "I'm unable to process your request because the Anthropic API key is not set. "
                        "Please set your ANTHROPIC_API_KEY environment variable or add it to a .env file. "
                        "You can get an API key at https://console.anthropic.com/"
                    ),
                    'success': False,
                    'error_type': 'api_key_missing'
                }
            else:
                return {
                    'response': f"I'm unable to process your request due to an initialization error: {error_msg}",
                    'success': False,
                    'error_type': 'initialization_error'
                }
        
        # Initialize the RAG pipeline
        rag_pipeline = RAGPipeline(
            llm_manager=llm_manager,
            db_connector=db_connector,
            table_name="ucla_player_stats"
        )
        
        # Process using the full RAG system
        result = rag_pipeline.process_query(user_query)
        
        # Clean up resources
        db_connector.close()
        
        # Return the response from the intelligent system
        if result.get('success', False):
            return {
                'response': result['response'],
                'success': True
            }
        else:
            return {
                'response': result.get('response', 'I could not process that query.'),
                'success': False,
                'error_type': result.get('error_type', 'unknown_error')
            }
        
    except Exception as e:
        logger.error(f"RAG pipeline error: {str(e)}")
        traceback.print_exc()
        return {
            'response': (
                "I encountered an error processing your question. "
                "Please try again or rephrase your question. "
                f"Error details: {str(e)}"
            ),
            'success': False,
            'error_type': 'pipeline_error'
        }


@app.route('/health')
def health():
    """
    Health check endpoint for monitoring application status.
    
    Returns:
        json: Health status and system information
    """
    try:
        conn = get_thread_safe_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ucla_player_stats")
        count = cursor.fetchone()[0]
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'records': count,
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/stats')
def stats():
    """
    Get application and database statistics.
    
    Returns:
        json: Various application metrics and database stats
    """
    try:
        conn = get_thread_safe_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(DISTINCT game_date) FROM ucla_player_stats")
        games_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT Name) FROM ucla_player_stats WHERE Name NOT IN ('Totals', 'TM')")
        players_count = cursor.fetchone()[0]
        
        return jsonify({
            'total_tokens': session.get('token_count', 0),
            'chat_sessions': len(session.get('chat_history', [])),
            'games_in_db': games_count,
            'players_tracked': players_count,
            'rag_status': 'active'
        })
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({
            'total_tokens': session.get('token_count', 0),
            'chat_sessions': len(session.get('chat_history', [])),
            'games_in_db': 0,
            'players_tracked': 0,
            'rag_status': 'error'
        })


@app.route('/history')
def history():
    """
    Get chat history for the current session.
    
    Returns:
        json: List of chat history entries
    """
    return jsonify(session.get('chat_history', []))


@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    """
    Clear chat history and reset session counters.
    
    Returns:
        json: Success status
    """
    session['chat_history'] = []
    session['token_count'] = 0
    return jsonify({'success': True})


if __name__ == '__main__':
    try:
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        
        # Test database connection
        conn = sqlite3.connect('data/ucla_wbb.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ucla_player_stats")
        record_count = cursor.fetchone()[0]
        conn.close()
        
        # Startup messages
        print("=" * 60)
        print("🏀 UCLA Women's Basketball RAG Analytics Web App")
        print("=" * 60)
        print(f"✅ Database connected: {record_count} records found")
        print(f"✅ Thread-safe connections implemented")
        print(f"🚀 Starting server on http://localhost:5001")
        print("=" * 60)
        
        # Start the Flask application
        app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
        
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        traceback.print_exc()
        logger.error(f"Application startup failed: {str(e)}") 