#Imports

import os
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from uuid import uuid4
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

#Initialization

ASTRA_DB_API_ENDPOINT = "" #insert endpoint 
ASTRA_DB_APPLICATION_TOKEN = "" #insert token

#Functions 

#Row to text functions

def row_to_text(row):
    return f"This data is about the player {row['personName']} (first name: {row["person_firstname"]}, last name: {row["person_lastname"]}), whose player id is {row['personId']}, during the basketball matchup {row['matchup']}. The game was played in the season of {row['season_year']}, precisely it was played on the date {row['game_date']} in the year {row['year']}. The id of the player's team is {row['teamId']}, the name of his team is {row['teamName']}, his team's tricode is {row['teamTricode']} and the city of his team is {row['teamCity']}. The opponent is {row['opponent']} and its tricode is {row['matchup'][-3:]}. {row['personName']}'s position on the court was {row['position']}, his jersey number was {row['jerseyNum']}. Player ided {row['personId']} played {row['minutes']} minutes in the {row['matchup']} game against {row['opponent']}. {row['personName']} attempted {row['fieldGoalsAttempted']} field goals, made {row['fieldGoalsMade']} of them, obtaining a percentage of {row['fieldGoalsPercentage']} in the {row['matchup']} game on {row['game_date']}. {row['personName']} attempted {row['threePointersAttempted']} field goals, made {row['threePointersMade']} of them for {row['teamSlug']}, against {row['matchup'][-3:]} obtaining a percentage of {row['threePointersPercentage']} in the {row['matchup']} game on {row['game_date']}. {row['personName']} attempted {row['freeThrowsAttempted']} field goals, made {row['freeThrowsMade']} of them for {row['teamSlug']}, obtaining a percentage of {row['freeThrowsPercentage']} in the {row['matchup']} game on {row['game_date']}. Player ided {row['personId']} won {row['reboundsOffensive']} offensive rebounds and won {row['reboundsDefensive']} defensive rebounds, for a total of {row['reboundsTotal']}. {row['personName']} made {row['assists']} assists for {row['teamTricode']} in {row['matchup']} game in the {row['season_year']} season against {row['matchup'][-3:]}, {row['steals']} steals, {row['blocks']} blocks, {row['turnovers']} turnovers, {row['foulsPersonal']} personal fouls, scored {row['points']} points. An additional comment: {row['comment']}." 

def row_to_text_2(row):
    return (
        f"{row['personName']} (#{row['jerseyNum']}, {row['position']}) "
        f"played for {row['teamName']} ({row['teamTricode']}) "
        f"against {row['opponent']} on {row['game_date']} in the year {row['year']} "
        f"({row['season_year']} season). "
        f"Minutes: {row['minutes']}, Points: {row['points']}, "
        f"FGM/FGA: {row['fieldGoalsMade']}/{row['fieldGoalsAttempted']} "
        f"({row['fieldGoalsPercentage']} FG%), "
        f"3PM/3PA: {row['threePointersMade']}/{row['threePointersAttempted']} "
        f"({row['threePointersPercentage']} 3P%), "
        f"FTM/FTA: {row['freeThrowsMade']}/{row['freeThrowsAttempted']} "
        f"({row['freeThrowsPercentage']} FT%). "
        f"Rebounds (Off/Def/Total): {row['reboundsOffensive']}/{row['reboundsDefensive']}/{row['reboundsTotal']}, "
        f"Assists: {row['assists']}, Steals: {row['steals']}, "
        f"Blocks: {row['blocks']}, Turnovers: {row['turnovers']}, "
        f"Fouls: {row['foulsPersonal']}. "
        f"Comment: {row['comment']}"
    )

#try joining

def row_to_text3(row):
    return ", ".join([f"{col}:{row[col]}" for col in row.index])

#Add documents to vector store

def embed_and_store(doc, vector_store):
    try:
        vector_store.add_documents(
            documents=[doc],  
            ids=[str(uuid4())]
        )
        return True
    except Exception as e:
        print(f"Failed to embed/store doc {doc.metadata}: {e}")
        return False

def parallel_upload_docs(documents, vector_store, max_workers=3):
    print("Embedding and uploading documents in parallel...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_and_store, doc, vector_store): doc for doc in documents}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Uploading docs"):
            result = future.result()  
            if result:
                print(f"Successfully uploaded doc: {future.result()}")
            else:
                print(f"Failed to upload doc: {future.result()}")

#Load data and transform it for LLM optimal use

csv_file_path = os.path.join(os.path.dirname(__file__), "clean_oneseason.csv")
df = pd.read_csv(csv_file_path)

documents = [
    Document(
        page_content=row_to_text(row),
        metadata={
            "game_date": row["game_date"], 
            "year": row["year"],
            "player": row["personName"], 
            "player_firstname": row["person_firstname"],
            "player_lastname": row["person_lastname"],
            "matchup": row["matchup"],
            "team": row["teamName"],
            "opponent": row['opponent']
        }
    )
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating documents")
]

#Embedding model
embeddings = OllamaEmbeddings(model="llama3.2:1b")

#Vector store: AstraDB
vector_store = AstraDBVectorStore(
    collection_name="RAG_tabular_4",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN
    )

parallel_upload_docs(documents, vector_store, 3)
