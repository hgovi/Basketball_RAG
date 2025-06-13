import logging
import re

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline for UCLA women's basketball data."""
    
    def __init__(self, llm_manager, db_connector, table_name="ucla_player_stats"):
        """Initialize pipeline with LLM manager and database connector.
        
        Args:
            llm_manager: LLM manager instance
            db_connector: Database connector instance
            table_name: Name of the table to query (default: ucla_player_stats)
        """
        self.llm = llm_manager
        self.db = db_connector
        self.table_name = table_name
        
        # Initialize components
        from src.entity_extractor import EntityExtractor
        from src.query_generator import SQLQueryGenerator
        
        self.entity_extractor = EntityExtractor(self.db, self.llm, table_name=self.table_name)
        self.query_generator = SQLQueryGenerator(self.llm, self.db, table_name=self.table_name)
        
        # Fallback strategies
        self.fallback_strategies = [
            self._simplify_aggregation_query,
            self._convert_to_basic_select,
            self._create_player_lookup_query,
        ]
    
    def process_query(self, user_query):
        """Process a natural language query and return response with comprehensive error handling."""
        logger.info(f"Processing query: {user_query}")
        
        try:
            # Step 1: Extract entities from query
            extracted_entities = self.entity_extractor.extract_entities(user_query)
            logger.info(f"Extracted entities: {extracted_entities}")
            
            # Step 2: Generate SQL query
            sql_query = self.query_generator.generate_sql_query(user_query, extracted_entities)
            logger.info(f"Generated SQL: {sql_query}")
            
            if not sql_query:
                return self._create_error_response(
                    user_query, 
                    "Failed to generate SQL query",
                    "The system could not create a valid SQL query for your request."
                )
            
            # Step 3: Validate SQL
            is_valid, validation_error = self.query_generator.validate_sql(sql_query)
            if not is_valid:
                logger.error(f"SQL validation failed: {validation_error}")
                
                # Try fallback strategies
                fallback_result = self._try_fallback_strategies(user_query, extracted_entities, validation_error)
                if fallback_result:
                    return fallback_result
                
                return self._create_error_response(
                    user_query,
                    f"SQL validation failed: {validation_error}",
                    "The system generated an incompatible query. Please try rephrasing your question."
                )
            
            # Step 4: Execute SQL query
            if self.db.conn is None:
                self.db.connect()
            
            query_results, sql_error = self.db.execute_query(sql_query, return_error=True)
            
            if sql_error:
                logger.error(f"SQL execution error: {sql_error}")
                
                # Try fallback strategies
                fallback_result = self._try_fallback_strategies(user_query, extracted_entities, sql_error)
                if fallback_result:
                    return fallback_result
                
                return self._create_error_response(
                    user_query,
                    f"SQL execution failed: {sql_error}",
                    "There was an error running your query. Please try a simpler version of your question."
                )
            
            # Step 5: Check for empty results
            if not query_results or len(query_results) == 0:
                logger.warning(f"Query returned no results")
                
                # Try fallback for empty results
                fallback_result = self._handle_empty_results(user_query, extracted_entities, sql_query)
                if fallback_result:
                    return fallback_result
                
                return self._create_empty_response(user_query, sql_query)
            
            # Step 6: Generate natural language response
            response = self._generate_response(user_query, sql_query, query_results)
            
            logger.info(f"Successfully processed query with {len(query_results)} results")
            
            return {
                "user_query": user_query,
                "extracted_entities": extracted_entities,
                "sql_query": sql_query,
                "query_results": query_results,
                "response": response,
                "success": True
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error processing query: {error_details}")
            
            return self._create_error_response(
                user_query,
                str(e),
                "I encountered an unexpected error while processing your request. Please try again or rephrase your question."
            )
    
    def _try_fallback_strategies(self, user_query, extracted_entities, original_error):
        """Try multiple fallback strategies when the main query fails."""
        logger.info(f"Trying fallback strategies for failed query")
        
        for i, strategy in enumerate(self.fallback_strategies):
            try:
                logger.info(f"Attempting fallback strategy {i+1}: {strategy.__name__}")
                
                fallback_sql = strategy(user_query, extracted_entities)
                if not fallback_sql:
                    continue
                
                # Validate fallback query
                is_valid, validation_error = self.query_generator.validate_sql(fallback_sql)
                if not is_valid:
                    logger.warning(f"Fallback strategy {i+1} generated invalid SQL: {validation_error}")
                    continue
                
                # Execute fallback query
                results, error = self.db.execute_query(fallback_sql, return_error=True)
                if error:
                    logger.warning(f"Fallback strategy {i+1} execution failed: {error}")
                    continue
                
                if results:
                    logger.info(f"Fallback strategy {i+1} succeeded with {len(results)} results")
                    response = self._generate_response(user_query, fallback_sql, results)
                    
                    return {
                        "user_query": user_query,
                        "extracted_entities": extracted_entities,
                        "sql_query": fallback_sql,
                        "query_results": results,
                        "response": response + "\n\n(Note: This answer uses a simplified query due to the complexity of your original request.)",
                        "success": True,
                        "fallback_used": True,
                        "original_error": original_error
                    }
                
            except Exception as e:
                logger.warning(f"Fallback strategy {i+1} failed: {str(e)}")
                continue
        
        logger.warning("All fallback strategies failed")
        return None
    
    def _simplify_aggregation_query(self, user_query, extracted_entities):
        """Fallback: Create a simpler aggregation query."""
        try:
            # Extract player names if mentioned
            player_filter = ""
            if extracted_entities and extracted_entities.get("player_names"):
                names = extracted_entities["player_names"]
                if isinstance(names, list):
                    name_list = "', '".join(names)
                    player_filter = f"AND Name IN ('{name_list}')"
                else:
                    player_filter = f"AND Name = '{names}'"
            
            # Create basic aggregation query
            if ("average" in user_query.lower() or "avg" in user_query.lower()):
                if ("team average" in user_query.lower() or "UCLA average" in user_query.lower()) and ("points" in user_query.lower()):
                    # Return the true UCLA WBB season average (using the "Totals" row) for points per game.
                    return f"""
                    SELECT ROUND(AVG(Pts), 2) AS team_avg_points
                     FROM {self.table_name}
                     WHERE Name = 'Totals'
                    """
                elif ("points" in user_query.lower()):
                    return f"""
                    SELECT Name, ROUND(AVG(Pts), 2) as avg_points
                     FROM {self.table_name}
                     WHERE Name NOT IN ('Totals', 'TM', 'Team') {player_filter}
                     GROUP BY Name
                     ORDER BY avg_points DESC
                     LIMIT 10
                    """
                elif ("rebounds" in user_query.lower()):
                    return f"""
                    SELECT Name, ROUND(AVG(Reb), 2) as avg_rebounds
                     FROM {self.table_name}
                     WHERE Name NOT IN ('Totals', 'TM', 'Team') {player_filter}
                     GROUP BY Name
                     ORDER BY avg_rebounds DESC
                     LIMIT 10
                    """
            elif ("total" in user_query.lower() or "sum" in user_query.lower()):
                if ("points" in user_query.lower()):
                    return f"""
                    SELECT Name, SUM(Pts) as total_points
                     FROM {self.table_name}
                     WHERE Name NOT IN ('Totals', 'TM', 'Team') {player_filter}
                     GROUP BY Name
                     ORDER BY total_points DESC
                     LIMIT 10
                    """
        except Exception as e:
            logger.warning(f"Failed to create simplified aggregation query: {e}")
        
        return None
    
    def _convert_to_basic_select(self, user_query, extracted_entities):
        """Fallback: Convert to basic SELECT query."""
        try:
            # Basic player stats query
            if extracted_entities and extracted_entities.get("player_names"):
                names = extracted_entities["player_names"]
                if isinstance(names, list):
                    name_list = "', '".join(names)
                    where_clause = f"Name IN ('{name_list}')"
                else:
                    where_clause = f"Name = '{names}'"
                
                return f"""
                SELECT Name, Pts, Reb, Ast, "TO", Stl, Blk, Opponent, game_date
                FROM {self.table_name}
                WHERE {where_clause} AND Name NOT IN ('Totals', 'TM', 'Team')
                ORDER BY game_date DESC
                LIMIT 20
                """
            
            # General top performers query
            if "best" in user_query.lower() or "top" in user_query.lower():
                return f"""
                SELECT Name, AVG(Pts) as avg_points, AVG(Reb) as avg_rebounds, AVG(Ast) as avg_assists
                FROM {self.table_name}
                WHERE Name NOT IN ('Totals', 'TM', 'Team')
                GROUP BY Name
                ORDER BY avg_points DESC
                LIMIT 10
                """
                
        except Exception as e:
            logger.warning(f"Failed to create basic select query: {e}")
        
        return None
    
    def _create_player_lookup_query(self, user_query, extracted_entities):
        """Fallback: Create a simple player lookup query."""
        try:
            return f"""
            SELECT DISTINCT Name, COUNT(*) as games_played
            FROM {self.table_name}
            WHERE Name NOT IN ('Totals', 'TM', 'Team')
            GROUP BY Name
            ORDER BY games_played DESC
            LIMIT 15
            """
        except Exception as e:
            logger.warning(f"Failed to create player lookup query: {e}")
        
        return None
    
    def _handle_empty_results(self, user_query, extracted_entities, sql_query):
        """Handle queries that return empty results."""
        logger.info("Handling empty results with alternative approach")
        
        # Check if it's a player-specific query with no results
        if extracted_entities and extracted_entities.get("player_names"):
            # Try without player filter to see if data exists
            modified_query = re.sub(
                r"WHERE.*?Name.*?=.*?'[^']*'.*?AND",
                "WHERE",
                sql_query,
                flags=re.IGNORECASE
            )
            
            if modified_query != sql_query:
                results, error = self.db.execute_query(modified_query, return_error=True)
                if not error and results:
                    response = f"I couldn't find specific data for that player. Here's what I found instead:\n"
                    response += self._generate_response(user_query, modified_query, results[:5])
                    
                    return {
                        "user_query": user_query,
                        "sql_query": modified_query,
                        "query_results": results[:5],
                        "response": response,
                        "success": True,
                        "fallback_used": True
                    }
        
        return None
    
    def _create_error_response(self, user_query, error, user_message):
        """Create a standardized error response."""
        return {
            "user_query": user_query,
            "error": error,
            "sql_query": None,
            "query_results": None,
            "response": user_message,
            "success": False
        }
    
    def _create_empty_response(self, user_query, sql_query):
        """Create a response for queries that return no data."""
        return {
            "user_query": user_query,
            "sql_query": sql_query,
            "query_results": [],
            "response": "I couldn't find any data matching your criteria. Please try rephrasing your question or asking about different players or statistics.",
            "success": False,
            "empty_results": True
        }
    
    def _generate_response(self, user_query, sql_query, query_results):
        """Generate natural language response from query results."""
        if not query_results:
            return "I couldn't find any data matching your request."
        
        # Create prompt for response generation
        prompt = f"""
        Based on the following UCLA women's basketball statistics, provide a clear and informative answer to the user's question.
        
        User question: {user_query}
        
        SQL query used: {sql_query}
        
        Query results (showing up to 10 rows): {query_results[:10]}
        
        Instructions:
        - Provide a direct answer to the user's question
        - Include specific numbers and statistics from the data
        - Format the response in a clear, readable way
        - If comparing players, present the comparison clearly
        - Keep the response concise but informative
        - Don't mention the SQL query or technical details
        """
        
        try:
            response = self.llm.generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            # Fallback to basic data presentation
            return self._create_basic_response(query_results)
    
    def _create_basic_response(self, query_results):
        """Create a basic response when LLM generation fails."""
        if not query_results:
            return "No data found."
        
        if len(query_results) == 1:
            return f"I found one result: {query_results[0]}"
        else:
            return f"I found {len(query_results)} results. Here are the first few: {query_results[:3]}"