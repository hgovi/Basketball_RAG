# UCLA Women's Basketball RAG System

This is a Retrieval-Augmented Generation (RAG) system designed to answer questions about the UCLA Women's Basketball Team using both structured and unstructured data.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```python
from basketball_rag import BasketballRAG

# Initialize the system with your OpenAI API key
rag = BasketballRAG(openai_api_key="your-api-key-here")
```

## Usage

### Loading Data

1. For structured data (CSV/Excel):
```python
rag.load_structured_data("path/to/your/data.csv")  # or .xlsx
```

2. For unstructured data (text):
```python
# Single text
rag.load_unstructured_data("Game summary text...")

# Multiple texts
rag.load_unstructured_data(["Game summary 1", "Game summary 2", ...])
```

### Asking Questions

```python
# Ask questions about statistics
answer = rag.answer_query("What are the team's average points per game?")

# Ask questions about game summaries
answer = rag.answer_query("What happened in the last game against USC?")
```

## Features

- Handles both structured (statistical) and unstructured (narrative) data
- Automatically determines the best way to answer each query
- Uses semantic search for text-based queries
- Provides concise, accurate answers based on available data
- Optimized for low token usage while maintaining accuracy

## Notes

- The system will return "Insufficient data to answer accurately" if it cannot find relevant information
- For best results, provide both structured and unstructured data
- Structured queries work best with statistical keywords (stats, average, total, etc.) 