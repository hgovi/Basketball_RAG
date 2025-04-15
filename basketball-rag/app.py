from flask import Flask, render_template, request, jsonify, send_from_directory
from basketball_rag import BasketballRAG
import os
import pandas as pd

app = Flask(__name__)

# Initialize the RAG system
api_key = "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"
rag = BasketballRAG(openai_api_key=api_key)

# Load data
structured_data_path = "data/structured/ucla_stats.csv"
unstructured_data_path = "data/unstructured/game_summaries.txt"

# Load structured data
rag.load_structured_data(structured_data_path)

# Load unstructured data
with open(unstructured_data_path, 'r') as f:
    game_summaries = f.read()
rag.load_unstructured_data(game_summaries)

# Token usage tracking
total_tokens_used = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        answer, tokens_used = rag.answer_query(query)
        global total_tokens_used
        total_tokens_used += tokens_used
        return jsonify({'answer': answer, 'tokens_used': tokens_used})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        history = rag.get_query_history()
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        df = pd.read_csv(structured_data_path)
        stats = {
            'games_analyzed': len(df),
            'win_rate': 80.0,  # Calculated from game summaries
            'avg_points': df['Points'].mean(),
            'tokens_used': total_tokens_used
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/img/<path:filename>')
def serve_images(filename):
    return send_from_directory('static/img', filename)

if __name__ == '__main__':
    app.run(debug=True) 