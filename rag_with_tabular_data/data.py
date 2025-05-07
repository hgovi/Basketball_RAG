import pandas as pd
data = pd.read_csv("C:\\Users\\Utente\\Desktop\\Project\\BruinAnalytics\\rag_with_tabular_data\\clean_oneseason.csv", sep=",")

def move_column_after(df, col_to_move, after_col):
    cols = list(df.columns)
    cols.remove(col_to_move)
    insert_at = cols.index(after_col) + 1
    cols.insert(insert_at, col_to_move)
    return df[cols]


def extract_opponent(matchup):
    matchup = matchup.strip()  
    if "@ " in matchup:  
        return matchup.split('@')[-1].strip()[:3]  
    elif "vs. " in matchup:  
        return matchup.split('vs. ')[-1].strip()[:3]  
    return None

def tri_to_name(data, tricode_to_name):

    def get_name(row):
        opp = row["opponent_tricode"]
        name = tricode_to_name.get(opp)
        return name
    
    data["opponent_tricode"] = data["matchup"].apply(extract_opponent)
    
    data["opponent"] = data.apply(get_name, axis=1)

    return data

tricode_to_name = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", 
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
} 

tri_to_name(data, tricode_to_name)
data = data.drop("opponent_tricode", axis=1)
print(data.head(n=5))

#data.to_csv("C:\\Users\\Utente\\Desktop\\Project\\BruinAnalytics\\rag_with_tabular_data\\clean_oneseason.csv")


def full_name_to_name(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    if 'personName' not in df.columns:
        raise ValueError("'personName' column not found in the data.")

    df[['person_firstname', 'person_lastname']] = df['personName'].str.split(n=1, expand=True)

    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to: {output_csv_path}")

def extract_year(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    
    df['year'] = df['game_date'].dt.year

    df.to_csv(output_csv_path, index=False)
    print(f"Updated file saved to: {output_csv_path}")



extract_year("C:\\Users\\Utente\\Desktop\\Project\\BruinAnalytics\\rag_with_tabular_data\\clean_oneseason.csv", "C:\\Users\\Utente\\Desktop\\Project\\BruinAnalytics\\rag_with_tabular_data\\clean_oneseason.csv")