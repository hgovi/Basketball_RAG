import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from io import StringIO
import os  

csv_file_path = os.path.join(os.path.dirname(__file__), "clean_oneseason.csv")
with open(csv_file_path, "r") as f:
    CSV_DATA = f.read()

# Cache the embedding model to avoid loading it on every rerun.
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Cache the generation pipeline.
@st.cache_resource(show_spinner=False)
def load_generation_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=device)

def row_to_text(row):
    """
    Convert a DataFrame row into a text summary.
    Customize this function as needed for your CSV structure.
    """
    return ", ".join([f"{col}: {row[col]}" for col in row.index])

def build_faiss_index(embeddings):
    """
    Build a FAISS index using L2 (Euclidean) distance.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, embed_model, index, texts, k=3):
    """
    Given a query, generate its embedding, search the FAISS index, and retrieve the k most relevant text chunks.
    """
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    results = [texts[i] for i in indices[0] if i < len(texts)]
    return results

def main():
    st.title("NBA Boxscore RAG Model with Embedded Data")
    
    # Instead of file upload, load the embedded CSV data.
    # Show only a preview of the CSV data (first 5 lines)
    lines = CSV_DATA.strip().splitlines()
    preview = "\n".join(lines[:5])
    st.markdown("Using built-in NBA boxscore CSV data (preview):")
    st.code(preview)
    
    # Read the CSV data from the string.
    df = pd.read_csv(StringIO(CSV_DATA))
    
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Convert each row of the DataFrame into a text document.
    texts = df.apply(row_to_text, axis=1).tolist()

    st.markdown("### Building Vector Store")
    embed_model = load_embedding_model()
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    vector_index = build_faiss_index(embeddings)
    st.success("Embeddings and FAISS index created!")

    st.markdown("### Ask a Question")
    query = st.text_input("Enter your question about the boxscore data:", "")
    if query:
        retrieved_context = retrieve_relevant_chunks(query, embed_model, vector_index, texts, k=3)
        context_text = "\n".join(retrieved_context)
        prompt = f"Using the following NBA boxscore data context:\n{context_text}\nAnswer the following question: {query}\nAnswer:"
        
        st.markdown("#### Retrieved Context")
        st.text(context_text)
        st.markdown("#### Prompt to LLM")
        st.text(prompt)
        
        st.markdown("### Generating Answer")
        gen_pipeline = load_generation_pipeline()
        generated_response = gen_pipeline(prompt, max_length=200, do_sample=True, temperature=0.7)
        answer_text = generated_response[0]['generated_text']
        
        st.markdown("#### Answer")
        st.write(answer_text)

if __name__ == "__main__":
    main()


    
