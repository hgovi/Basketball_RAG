import sqlite3
import pandas as pd
import logging
import time
import re
from langchain_community.utilities import SQLDatabase

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """Handles database connections and operations for UCLA women's basketball data."""
    
    def __init__(self, db_path='data/ucla_wbb.db'):
        """Initialize with path to SQLite database (default: ucla_wbb.db)."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.langchain_db = None
        self.query_stats = {'total_queries': 0, 'failed_queries': 0, 'avg_execution_time': 0}
        
    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.langchain_db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            
            # Enable foreign keys and optimize SQLite settings
            self.cursor.execute("PRAGMA foreign_keys = ON")
            self.cursor.execute("PRAGMA cache_size = 10000")
            self.cursor.execute("PRAGMA temp_store = MEMORY")
            
            logger.info(f"Connected to database at {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.info("Database connection closed")
    
    def execute_query(self, query, return_error=False, validate_first=True):
        """Execute SQL query and return results with comprehensive error handling."""
        if not self.conn:
            self.connect()
        
        start_time = time.time()
        self.query_stats['total_queries'] += 1
        
        try:
            # Optional pre-validation
            if validate_first:
                validation_error = self._validate_query_syntax(query)
                if validation_error:
                    logger.error(f"Query validation failed: {validation_error}\nQuery: {query}")
                    self.query_stats['failed_queries'] += 1
                    if return_error:
                        return None, f"Validation error: {validation_error}"
                    return None
            
            logger.debug(f"Executing query: {query}")
            
            # Execute with timeout protection
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            
            execution_time = time.time() - start_time
            self._update_query_stats(execution_time)
            
            logger.info(f"Query executed successfully in {execution_time:.3f}s, returned {len(result)} rows")
            
            if return_error:
                return result, None
            return result
            
        except sqlite3.Error as e:
            execution_time = time.time() - start_time
            self.query_stats['failed_queries'] += 1
            
            error_msg = f"SQLite error: {str(e)}"
            logger.error(f"Error executing query: {error_msg}\nQuery: {query}\nExecution time: {execution_time:.3f}s")
            
            if return_error:
                return None, error_msg
            return None
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.query_stats['failed_queries'] += 1
            
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Error executing query: {error_msg}\nQuery: {query}\nExecution time: {execution_time:.3f}s")
            
            if return_error:
                return None, error_msg
            return None
    
    def test_query_execution(self, query):
        """Test query execution and return detailed results for diagnostics."""
        test_result = {
            'query': query,
            'syntax_valid': False,
            'execution_successful': False,
            'result_count': 0,
            'execution_time': 0,
            'error_message': None,
            'result_sample': None
        }
        
        # Test syntax validation
        syntax_error = self._validate_query_syntax(query)
        if syntax_error:
            test_result['error_message'] = f"Syntax error: {syntax_error}"
            return test_result
        
        test_result['syntax_valid'] = True
        
        # Test execution
        start_time = time.time()
        try:
            results, error = self.execute_query(query, return_error=True, validate_first=False)
            test_result['execution_time'] = time.time() - start_time
            
            if error:
                test_result['error_message'] = error
            else:
                test_result['execution_successful'] = True
                test_result['result_count'] = len(results) if results else 0
                test_result['result_sample'] = results[:3] if results else []
                
        except Exception as e:
            test_result['execution_time'] = time.time() - start_time
            test_result['error_message'] = f"Test execution failed: {str(e)}"
        
        return test_result
    
    def _validate_query_syntax(self, query):
        """Validate query syntax without executing it."""
        if not query or not query.strip():
            return "Empty query"
        
        # Check for basic SQL injection patterns
        dangerous_patterns = [
            r';\s*(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER)\s+',
            r'UNION\s+SELECT.*--',
            r'\/\*.*\*\/',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return f"Potentially dangerous SQL pattern detected"
        
        # Check for required elements
        if not re.search(r'\bSELECT\b', query, re.IGNORECASE):
            return "Query must contain SELECT statement"
        
        # Use SQLite's EXPLAIN to validate syntax
        try:
            explain_query = f"EXPLAIN {query}"
            self.cursor.execute(explain_query)
            return None  # No syntax errors
        except sqlite3.Error as e:
            return f"SQLite syntax error: {str(e)}"
        except Exception as e:
            return f"Syntax validation error: {str(e)}"
    
    def _update_query_stats(self, execution_time):
        """Update query execution statistics."""
        total = self.query_stats['total_queries']
        current_avg = self.query_stats['avg_execution_time']
        self.query_stats['avg_execution_time'] = ((current_avg * (total - 1)) + execution_time) / total
    
    def get_query_statistics(self):
        """Get query execution statistics for monitoring."""
        stats = self.query_stats.copy()
        stats['success_rate'] = (stats['total_queries'] - stats['failed_queries']) / max(stats['total_queries'], 1)
        return stats
    
    def get_table_schema(self, table_name="ucla_player_stats"):
        """Get schema for a table with enhanced error handling."""
        if not self.conn:
            self.connect()
            
        try:
            query = f"PRAGMA table_info({table_name})"
            result = self.execute_query(query, validate_first=False)
            
            if result:
                # Format as readable schema
                schema = []
                for col in result:
                    schema.append({
                        "name": col[1],
                        "type": col[2],
                        "notnull": col[3],
                        "pk": col[5]
                    })
                logger.info(f"Retrieved schema for table '{table_name}' with {len(schema)} columns")
                return schema
        except Exception as e:
            logger.error(f"Error retrieving schema for table '{table_name}': {str(e)}")
        
        logger.warning(f"Could not retrieve schema for table '{table_name}'")
        return None
    
    def get_distinct_values(self, column, table="ucla_player_stats", limit=1000):
        """Get distinct values for a column with error handling."""
        if not self.conn:
            self.connect()
            
        try:
            query = f'SELECT DISTINCT "{column}" FROM {table} WHERE "{column}" IS NOT NULL LIMIT {limit}'
            result = self.execute_query(query, validate_first=False)
            
            if result:
                values = [item[0] for item in result]
                logger.debug(f"Retrieved {len(values)} distinct values for '{column}' in table '{table}'")
                return values
        except Exception as e:
            logger.error(f"Error getting distinct values for '{column}': {str(e)}")
        
        logger.warning(f"No distinct values found for '{column}' in table '{table}'")
        return []
    
    def get_table_names(self):
        """Get all table names in the database."""
        if not self.conn:
            self.connect()
            
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            result = self.execute_query(query, validate_first=False)
            
            if result:
                tables = [item[0] for item in result]
                logger.info(f"Database contains {len(tables)} tables: {', '.join(tables)}")
                return tables
        except Exception as e:
            logger.error(f"Error retrieving table names: {str(e)}")
        
        return []
    
    def get_row_count(self, table_name):
        """Get the number of rows in a table."""
        if not self.conn:
            self.connect()
            
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            result = self.execute_query(query, validate_first=False)
            
            if result and result[0]:
                count = result[0][0]
                logger.info(f"Table '{table_name}' contains {count} rows")
                return count
        except Exception as e:
            logger.error(f"Error getting row count for '{table_name}': {str(e)}")
        
        return 0