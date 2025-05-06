# Basketball RAG System with Llama

A Retrieval-Augmented Generation (RAG) system for basketball data using Llama models locally instead of OpenAI's API.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Initialize the system

```python
from basketball_rag import BasketballRAG

rag = BasketballRAG(
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    device="cpu"  # Use "cuda" for GPU
)
```

### Load data

```python
# Load structured data (CSV/Excel)
rag.load_structured_data("path/to/data.csv")

# Load unstructured data (text)
rag.load_unstructured_data("Game summary text...")
```

### Ask questions

```python
# Query the system
answer, tokens_used = rag.answer_query("What are the team's average points per game?")
print(answer)
```

## Model Options

### Gated Models (Require Authentication)
- `meta-llama/Llama-2-7b-chat-hf` (requires Hugging Face account & approval)
- `meta-llama/Llama-2-13b-chat-hf` (requires Hugging Face account & approval)

### Open-Access Models (No Authentication Required)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (small, fast, no login needed)
- `facebook/opt-350m` (very small model for testing)
- `mistralai/Mistral-7B-Instruct-v0.2` (good performance, freely available)

## Authentication

To use gated models like Llama-2:

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co/join)
2. Request access to Llama 2 at [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
3. Log in via CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

## Running the demo

```bash
# Interactive mode
python demo.py --interactive

# With specific options
python demo.py --model="meta-llama/Llama-2-7b-chat-hf" --device="cuda"
```

## Web application

```bash
python app.py --port=8080 --model="meta-llama/Llama-2-7b-chat-hf"
```

## Testing

```bash
python test_script.py