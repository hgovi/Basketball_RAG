#Imports

import os
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from langchain_community.llms import Ollama  
from uuid import uuid4
import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

ASTRA_DB_API_ENDPOINT = "" #insert endpoint
ASTRA_DB_APPLICATION_TOKEN = "" #insert token

def find_metadata(query, llm):
    prompt = (
        f"You are extracting structured metadata from the following user query. "
        f"Output ONLY a JSON object. DO NOT write explanations, headers, or anything else.\n\n"
        
        f"User query:\n\"{query}\"\n\n"

        f"Extract and return the following fields:\n"
        f"- game_date (format: \"YYYY-MM-DD\")\n"
        f"- year (4-digit string, e.g., \"2023\")\n"
        f"- player (full name if available, e.g., \"LeBron James\")\n"
        f"- person_firstname\n"
        f"- person_lastname\n"
        f"- opponent (must match one of the valid team names listed below)\n"
        f"- matchup (e.g., \"CHA @ LAL\")\n"
        f"- team (the team the player plays for)\n\n"

        f"If a value is not available or cannot be confidently inferred from the query, use None.\n"
        f"DO NOT guess values. Only fill a field if you are confident it is present in the query.\n\n"

        f"Important instructions:\n"
        f"- If you can extract a full date (day/month/year), fill in 'game_date' as a string in the format YYYY-MM-DD and also fill 'year'.\n"
        f"- If you can only determine the year, fill in 'year' and set 'game_date' to None.\n"
        f"- If the query only mentions a first name (e.g., \"LeBron\"), set 'person_firstname' to that value, and set 'player' and 'person_lastname' to None.\n"
        f"- Use exact team names for 'opponent' (from the list below). If the opponent is not mentioned or does not match, set it to None.\n"
        f"- Only fill in the 'team' (the team the player plays for) if it is clearly stated or implied. Otherwise, set to None.\n\n"

        f"Valid NBA teams (for 'opponent' field):\n"
        f"Atlanta Hawks, Boston Celtics, Brooklyn Nets, Charlotte Hornets, Chicago Bulls, Cleveland Cavaliers, Dallas Mavericks, Denver Nuggets, Detroit Pistons, Golden State Warriors, Houston Rockets, Indiana Pacers, Los Angeles Clippers, Los Angeles Lakers, Memphis Grizzlies, Miami Heat, Milwaukee Bucks, Minnesota Timberwolves, New Orleans Pelicans, New York Knicks, Oklahoma City Thunder, Orlando Magic, Philadelphia 76ers, Phoenix Suns, Portland Trail Blazers, Sacramento Kings, San Antonio Spurs, Toronto Raptors, Utah Jazz, Washington Wizards.\n\n"

        f"Example output:\n"
        f"Query: 'How many assists did LeBron James collect on January 3rd 2023 against the Charlotte Hornets?'"
        f"{{\n"
        f"  \"game_date\": \"2023-01-03\",\n"
        f"  \"year\": \"2023\",\n"
        f"  \"player\": \"LeBron James\",\n"
        f"  \"person_firstname\": \"LeBron\",\n"
        f"  \"person_lastname\": \"James\",\n"
        f"  \"opponent\": \"Charlotte Hornets\",\n"
        f"  \"matchup\": \"None\",\n"
        f"  \"team\": \"None\"\n"
        f"}}\n"
    )
    print(query)
    response = llm.invoke(prompt)

    try:
        metadata = json.loads(response)
        print(metadata)
    except json.JSONDecodeError:
        print("⚠️ Warning: LLM did not return valid JSON. Raw response:")
        print(response)
        return None
    
    expected_keys = {"game_date", "year", "player", "person_firstname", "person_lastname", "opponent", "matchup", "team"}
    for key in expected_keys:
        if key not in metadata:
            metadata[key] = None  

    return metadata

def query_model(query, vector_store):
    try:
        results = vector_store.similarity_search_with_score(
            query=query, k=30
        )
        outputs = []
        for doc, score in results:
            outputs.append((doc.page_content, score))
        return outputs
    except Exception as e:
        print(f"Failed query", query, {e})
        return []

def parallel_query(vector_store, queries):
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(query_model, query, vector_store): query for query in queries}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    return results

def query_with_filter(vector_store, query, k=10):
    llm = Ollama(model="llama3:latest")
    try:
        filter = find_metadata(query, llm)
        if filter is not None:
            filter = {k: v for k, v in filter.items() if v is not None}
        print(filter)

        results = vector_store.similarity_search_with_score(query=query, filter=filter, k=k)
        print("results: ", results)
        outputs = []
        
        for doc, score in results:
            outputs.append((doc.page_content, score))
        print("outputs: ", outputs)
        return outputs
    
    except Exception as e:
        print("Failed query ", {query}, ":", {e})
        return []

def parallel_query_with_filter(vector_store, queries, k=5):
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(query_with_filter, vector_store, query, k): query for query in queries
        }
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                results.append(result)
    return results

def save_retrieval_results_to_csv(results_list, queries, path='C:\\Users\\Utente\\Desktop\\Project\\BruinAnalytics\\rag_with_tabular_data\\retrieval_results_comparison2.csv'):
    
    if not results_list or not queries:
        print("⚠️ One of the input lists is empty!")
        return

    rows = []

    for i, query in enumerate(queries):
        hits_for_models = [results_list[j][i] if i < len(results_list[j]) else [("", None)] for j in range(len(results_list))]

        max_hits = max(len(hits) for hits in hits_for_models)
        
        for hits in hits_for_models:
            while len(hits) < max_hits:
                hits.append(("", None))

        for row_index in range(max_hits):
            row = {"query_text": query}
            for model_idx, hits in enumerate(hits_for_models):
                print("hits: ", hits[row_index])
                text, score = hits[row_index]
                row[f"text_model{model_idx+1}"] = text
                row[f"score_model{model_idx+1}"] = score
            rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(path, index=False, sep=";")
    print(f"✅ Results saved to {path}")

#Testing

embeddings = OllamaEmbeddings(model="llama3.2:1b")
llm = Ollama(model="llama3.2:1b")

vector_store_1 = AstraDBVectorStore(
    collection_name="RAG_tabular",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

vector_store_2 = AstraDBVectorStore(
    collection_name="RAG_tabular_2",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

vector_store_3 = AstraDBVectorStore(
    collection_name="RAG_tabular_3",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

vector_store_4 = AstraDBVectorStore(
    collection_name="RAG_tabular_4",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

'''
test_queries = ["How many points did Davis Bertans score against the Lakers in 2023?",
                "In what position does Gordon Hayward play in 2024?",
                "Did Vasilije Micic play against Atlanta on January third?",
                "How many rebounds did Kenrich Williams complete against Philadelphia?"]

results1 = parallel_query(vector_store_1, test_queries)
results2 = parallel_query(vector_store_2, test_queries)

results = [results1, results2]

save_retrieval_results_to_csv(results, test_queries)
'''
# We do not get optimal results. Problem is with this structure of textualized rows, all row appear very similar to LLM model which returns similar vectors, so similarity
#check retrieves many vectors with very similar score.

## We either increase number of vectors retrieved to minimize error, but this would mean more context and more expensive queries in next stage, or changing row to text structure (more concise means more unique vector), 
## or  use metadata filtering (should be done in advance through NLP)
# We could combine, so metadata filtering does first stage of narrowing down to lets say the one right player, and similarity search focuses on game and period.
##To perform metadata filtering, we could use further LLM calls to prepare a code that included metadata filtering based on query. 
##Another approach would see a change in the LLM as a more performing LLM could result in more detailed embeddings while a small and fast model like the one we are using could give more high-level embeddings


#Testing metadata function 
test_queries = ["How many points did Davis Bertans score against the Lakers in 2023?",]
              #  "In what position does Gordon Hayward play in 2024?"]
'''
                "Did Vasilije Micic play against Atlanta on January third?",
                "How many rebounds did Kenrich Williams complete against Philadelphia?",
                "Please tell me Gilgeous's field goals attempts against Utah Jazz",
                ]
'''
results = parallel_query_with_filter(vector_store_4, test_queries)

save_retrieval_results_to_csv(results, test_queries)
