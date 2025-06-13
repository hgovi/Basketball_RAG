import re
import json
from thefuzz import process
from langchain_core.prompts import PromptTemplate

class EntityExtractor:
    """Extract and resolve entities from user queries for UCLA women's basketball data."""
    
    def __init__(self, db_connector, llm_manager, table_name="ucla_player_stats"):
        """Initialize with database connector and LLM manager.
        
        Args:
            db_connector: Database connector instance
            llm_manager: LLM manager instance
            table_name: Name of the table to query (default: ucla_player_stats)
        """
        self.db = db_connector
        self.llm = llm_manager
        self.table_name = table_name
        self.entity_cache = {}  # Cache for entity resolution results
        self.dataset_type = "ucla"
        
        # Pre-load common entities for faster matching
        self._load_common_entities()
    
    def _load_common_entities(self):
        """Load common entities from UCLA women's basketball database."""
        # Connect to database if not already connected
        if self.db.conn is None:
            self.db.connect()
            
        # UCLA women's basketball specific entities
        self.players = self.db.get_distinct_values("Name", table=self.table_name)
        self.player_numbers = self.db.get_distinct_values("No", table=self.table_name)
        self.opponents = self.db.get_distinct_values("Opponent", table=self.table_name)
        self.teams = ["UCLA"]  # Only one team in this dataset
        self.seasons = ["2024-2025"]  # Only one season in this dataset
    
    def extract_entities(self, query):
        """Extract entities from query using LLM."""
        # Create prompt for UCLA women's basketball
        prompt = f"""
        Extract entities from this UCLA women's basketball statistics query.
        Return a JSON object with these fields:
        - player_names: Array of player names mentioned (can be multiple players)
        - player_number: Jersey number mentioned (if any)
        - opponent: Opponent team mentioned (if any)
        - statistic: Specific statistic mentioned (points, rebounds, assists, etc.)
        - comparison: Any comparison operators (>, <, =, etc.)
        - value: Any numeric value mentioned for comparison
        - exclude_totals: Set to true if the query mentions excluding team totals or only individual players
        - is_comparison_query: Set to true if the query is asking to compare multiple players
        
        Query: {query}
        
        JSON output:
        """
        
        # Generate extraction using LLM
        result = self.llm.generate_text(prompt)
        
        # Parse the JSON from the response
        try:
            # Find the JSON part in the response
            json_match = re.search(r'({.*})', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                entities = json.loads(json_str)
            else:
                # Fallback to matching with pattern extraction
                entities = self._pattern_extract_entities(query)
        except Exception as e:
            print(f"Error parsing JSON from LLM: {str(e)}")
            # Fallback to pattern extraction
            entities = self._pattern_extract_entities(query)
        
        # Resolve and validate entities
        return self._resolve_entities(entities)
    
    def _pattern_extract_entities(self, query):
        """Fallback extraction using regex patterns."""
        if self.dataset_type == "ucla":
            entities = {
                "player_name": None,
                "player_number": None,
                "opponent": None,
                "statistic": None,
                "comparison": None,
                "value": None
            }
            
            # Try to extract player numbers (like #7 or No. 51)
            number_pattern = r'#(\d+)|No\. (\d+)|number (\d+)'
            number_match = re.search(number_pattern, query, re.IGNORECASE)
            if number_match:
                # Get the first non-None group
                for group in number_match.groups():
                    if group is not None:
                        entities["player_number"] = group
                        break
        else:
            entities = {
                "player_name": None,
                "team_name": None,
                "opponent": None,
                "season": None,
                "statistic": None,
                "comparison": None,
                "value": None
            }
            
            # Extract seasons (like 2023-24)
            season_pattern = r'(\d{4}-\d{2,4}|\d{4}[-/]\d{2,4})'
            season_match = re.search(season_pattern, query)
            if season_match:
                entities["season"] = season_match.group(1)
        
        # Extract statistics (common for both datasets)
        stat_keywords = ["points", "rebounds", "assists", "steals", "blocks", "turnovers", 
                        "pts", "reb", "ast", "stl", "blk", "to", "field goals", "three pointers", "free throws"]
        for stat in stat_keywords:
            if stat in query.lower():
                # Map abbreviations to full names
                stat_mapping = {
                    "pts": "points", "reb": "rebounds", "ast": "assists",
                    "stl": "steals", "blk": "blocks", "to": "turnovers"
                }
                entities["statistic"] = stat_mapping.get(stat, stat)
                break
        
        # Extract comparisons
        comparison_pattern = r'(more than|less than|at least|at most|equal to|>|<|>=|<=|=)'
        comparison_match = re.search(comparison_pattern, query)
        if comparison_match:
            entities["comparison"] = comparison_match.group(1)
        
        # Extract numeric values
        value_pattern = r'\b(\d+)\b'
        value_match = re.search(value_pattern, query)
        if value_match:
            entities["value"] = value_match.group(1)
        
        return entities
    
    def _resolve_entities(self, entities):
        """Resolve extracted entities to database entries using fuzzy matching."""
        resolved = {}
        
        # Handle player names - could be a single string or an array
        if entities.get("player_names"):
            player_names = entities["player_names"]
            # Convert to list if it's a string
            if isinstance(player_names, str):
                player_names = [player_names]
            
            # Resolve each player name
            resolved_names = []
            for name in player_names:
                player_match = self._fuzzy_match(name, self.players)
                if player_match:
                    resolved_names.append(player_match)
            
            if resolved_names:
                resolved["player_names"] = resolved_names
        
        # Handle player number
        if entities.get("player_number"):
            number_match = self._fuzzy_match(str(entities["player_number"]), self.player_numbers)
            if number_match:
                resolved["player_number"] = number_match
        
        # Resolve opponent
        if entities.get("opponent"):
            opponent_match = self._fuzzy_match(entities["opponent"], self.opponents)
            if opponent_match:
                resolved["opponent"] = opponent_match
        
        # Copy other fields directly
        for field in ["statistic", "comparison", "value", "exclude_totals", "is_comparison_query"]:
            if field in entities and entities[field] is not None:
                resolved[field] = entities[field]
        
        return resolved
    
    def _fuzzy_match(self, query, options, threshold=75):
        """Find the best match for a query in a list of options."""
        if not query or not options:
            return None
        
        # Ensure query is a string
        if not isinstance(query, str):
            print(f"Warning: Non-string query value: {query}, type: {type(query)}")
            return None
            
        # Check cache first
        try:
            cache_key = f"{query}:{','.join(str(opt) for opt in options[:5])}"
            if cache_key in self.entity_cache:
                return self.entity_cache[cache_key]
            
            # Find best match
            match, score = process.extractOne(query, options)
            
            # Only accept match if score is above threshold
            if score >= threshold:
                self.entity_cache[cache_key] = match
                return match
        except Exception as e:
            print(f"Error in fuzzy matching: {str(e)}")
            print(f"Query: {query}, type: {type(query)}")
            print(f"Options sample: {options[:3] if options else 'None'}")
        
        return None