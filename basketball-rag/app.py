"""
Enhanced Basketball RAG Web Application (Llama Version)

This Flask application provides a web interface for the Enhanced Basketball RAG system,
allowing users to interact with both structured and unstructured basketball data
using an open-source Llama model instead of OpenAI.
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from basketball_rag import BasketballRAG
import pandas as pd
import argparse
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure app with command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run the Enhanced Basketball RAG Web Application')
    parser.add_argument('--structured-data', type=str, default='data/structured/ucla_stats.csv',
                        help='Path to structured data file (CSV or Excel)')
    parser.add_argument('--unstructured-data', type=str, default='data/unstructured/game_summaries.txt',
                        help='Path to unstructured data file (text)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to run the Flask application on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Hugging Face model name to use')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run models on (cpu or cuda)')
    return parser.parse_args()

# Initialize the RAG system
rag = None
# Token usage tracking
total_tokens_used = 0

def load_data(args):
    """Load data into the RAG system"""
    global rag
    
    # Initialize the RAG system with Llama model
    logging.info(f"Initializing RAG system with model {args.model} on {args.device}")
    rag = BasketballRAG(
        llm_model=args.model,
        device=args.device,
    )
    
    # Load structured data
    if os.path.exists(args.structured_data):
        logging.info(f"Loading structured data from {args.structured_data}")
        rag.load_structured_data(args.structured_data)
    else:
        app.logger.warning(f"Structured data file not found: {args.structured_data}")
    
    # Load unstructured data
    if os.path.exists(args.unstructured_data):
        logging.info(f"Loading unstructured data from {args.unstructured_data}")
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
        
        # Get query analysis
        query_analysis = rag._analyze_query(query)
        
        # Process the query through the RAG system
        answer, tokens_used = rag.answer_query(query, use_history=use_history)
        
        # Update token usage
        global total_tokens_used
        total_tokens_used += tokens_used
        
        # Return enhanced response with query analysis
        return jsonify({
            'answer': answer,
            'tokens_used': tokens_used,
            'total_tokens_used': total_tokens_used,
            'query_analysis': query_analysis
        })
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-query', methods=['POST'])
def analyze_query():
    """Analyze a query without executing it fully"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Get query analysis
        query_analysis = rag._analyze_query(query)
        
        return jsonify({
            'query_analysis': query_analysis
        })
    except Exception as e:
        app.logger.error(f"Error analyzing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/data-preview', methods=['GET'])
def get_data_preview():
    """Get a preview of the loaded structured data"""
    try:
        if rag.structured_data is not None:
            # Get basic info
            columns = rag.structured_data.columns.tolist()
            sample_data = rag.structured_data.head(5).to_dict(orient='records')
            data_info = {
                'row_count': len(rag.structured_data),
                'column_count': len(columns),
                'columns': columns,
                'sample_data': sample_data
            }
            
            return jsonify(data_info)
        else:
            return jsonify({'error': 'No structured data loaded'}), 404
    except Exception as e:
        app.logger.error(f"Error getting data preview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/execute-calculation', methods=['POST'])
def execute_calculation():
    """Execute a specific calculation on the structured data"""
    data = request.json
    calculation = data.get('calculation', '')
    parameters = data.get('parameters', {})
    
    if not calculation:
        return jsonify({'error': 'No calculation specified'}), 400
    
    try:
        # Ensure rag has stat functions
        if not hasattr(rag, 'stat_functions') or not rag.stat_functions:
            return jsonify({'error': 'Statistical functions not available'}), 500
        
        # Check if calculation exists
        if calculation not in rag.stat_functions:
            return jsonify({'error': f'Unknown calculation: {calculation}'}), 400
        
        # Execute the calculation
        result = None
        
        if calculation == 'average':
            if 'column' in parameters:
                result = float(rag.stat_functions[calculation](rag.structured_data, parameters['column']))
        elif calculation == 'correlation':
            if 'column1' in parameters and 'column2' in parameters:
                result = float(rag.stat_functions[calculation](
                    rag.structured_data, parameters['column1'], parameters['column2']
                ))
        # Add more calculation types as needed
        
        if result is None:
            return jsonify({'error': 'Could not execute calculation with provided parameters'}), 400
        
        return jsonify({
            'calculation': calculation,
            'parameters': parameters,
            'result': result
        })
    except Exception as e:
        app.logger.error(f"Error executing calculation: {str(e)}")
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
    """Get enhanced statistics about the data and system usage"""
    try:
        stats = {
            'games_analyzed': 0,
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
            
            # Add more advanced stats
            # Points per minute if available
            if 'PTS_PER_MIN' in df.columns:
                stats['pts_per_min'] = df['PTS_PER_MIN'].mean()
            
            # Shooting efficiency if available
            if 'FG_PCT' in df.columns:
                stats['fg_pct'] = df['FG_PCT'].mean() * 100
            if 'FG3_PCT' in df.columns:
                stats['3p_pct'] = df['FG3_PCT'].mean() * 100
        
        # Add unstructured data stats
        if hasattr(rag, 'unstructured_chunks') and rag.unstructured_chunks:
            stats['unstructured_chunks'] = len(rag.unstructured_chunks)
        
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/available-calculations', methods=['GET'])
def get_available_calculations():
    """Get list of available statistical calculations"""
    try:
        if hasattr(rag, 'stat_functions') and rag.stat_functions:
            calculations = list(rag.stat_functions.keys())
            
            # Add descriptions for common calculations
            calculation_info = {
                'average': 'Calculate the average (mean) of a numeric column',
                'sum': 'Calculate the sum of a numeric column',
                'min': 'Find the minimum value in a column',
                'max': 'Find the maximum value in a column',
                'median': 'Calculate the median (middle value) of a column',
                'correlation': 'Calculate the correlation coefficient between two columns',
                'count_above': 'Count values above a threshold',
                'count_below': 'Count values below a threshold',
                'filter_equals': 'Filter data where column equals value',
                'filter_greater': 'Filter data where column is greater than value',
                'filter_less': 'Filter data where column is less than value',
                'filter_between': 'Filter data where column is between min and max values',
                'group_avg': 'Average grouped by a column',
                'group_sum': 'Sum grouped by a column',
            }
            
            # Create response with available calculations and descriptions
            response = []
            for calc in calculations:
                calc_info = {'name': calc}
                if calc in calculation_info:
                    calc_info['description'] = calculation_info[calc]
                response.append(calc_info)
            
            return jsonify({'calculations': response})
        else:
            return jsonify({'calculations': []})
    except Exception as e:
        app.logger.error(f"Error getting available calculations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-history', methods=['GET'])
def download_history():
    """Save and return the query history as a JSON file"""
    try:
        # Ensure downloads directory exists
        os.makedirs('static/downloads', exist_ok=True)
        
        rag.save_query_history('static/downloads/query_history.json')
        return send_from_directory('static/downloads', 'query_history.json', as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error downloading history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-data-insights', methods=['GET'])
def download_data_insights():
    """Generate and download a comprehensive data insights report"""
    try:
        if rag.structured_data is None:
            return jsonify({'error': 'No structured data available'}), 404
        
        # Ensure downloads directory exists
        os.makedirs('static/downloads', exist_ok=True)
        
        # Generate insights
        df = rag.structured_data
        
        # Basic statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        basic_stats = df[numeric_cols].describe().to_dict()
        
        # Correlations
        correlations = df[numeric_cols].corr().to_dict()
        
        # Top performances
        top_performances = {}
        for col in numeric_cols:
            top_idx = df[col].nlargest(3).index.tolist()
            if 'Game' in df.columns and 'Opponent' in df.columns:
                top_performances[col] = [
                    {'game': int(df.loc[idx, 'Game']), 
                     'opponent': df.loc[idx, 'Opponent'], 
                     'value': float(df.loc[idx, col])}
                    for idx in top_idx if idx in df.index
                ]
        
        # Create insights report
        insights = {
            'report_date': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'columns_list': df.columns.tolist()
            },
            'basic_statistics': basic_stats,
            'correlations': correlations,
            'top_performances': top_performances
        }
        
        # Save insights to file
        with open('static/downloads/data_insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        return send_from_directory('static/downloads', 'data_insights.json', as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error generating data insights: {str(e)}")
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
    
    # Log data loading status
    if rag.structured_data is not None:
        app.logger.info(f"Loaded structured data with {len(rag.structured_data)} rows and {len(rag.structured_data.columns)} columns")
    
    if hasattr(rag, 'unstructured_chunks') and rag.unstructured_chunks:
        app.logger.info(f"Loaded unstructured data with {len(rag.unstructured_chunks)} chunks")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()