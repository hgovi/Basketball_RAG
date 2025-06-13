"""
Test suite for UCLA Basketball RAG Application
"""
import pytest
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from app import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def runner():
    """Create test runner"""
    return app.test_cli_runner()

class TestApplication:
    """Test application endpoints"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'database' in data
        assert 'records' in data
    
    def test_index_page(self, client):
        """Test main index page"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'UCLA' in response.data
    
    def test_stats_endpoint(self, client):
        """Test statistics endpoint"""
        response = client.get('/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'total_tokens' in data
        assert 'games_in_db' in data
    
    def test_query_endpoint_valid(self, client):
        """Test query endpoint with valid basketball question"""
        response = client.post('/query',
                             data=json.dumps({'query': 'Who is the top scorer?'}),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'response' in data
        assert 'tokens' in data
        assert len(data['response']) > 0
    
    def test_query_endpoint_empty(self, client):
        """Test query endpoint with empty query"""
        response = client.post('/query',
                             data=json.dumps({'query': ''}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_query_endpoint_usc(self, client):
        """Test specific USC query"""
        response = client.post('/query',
                             data=json.dumps({'query': 'Who scored the most points against USC?'}),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'USC' in data['response'] or 'points' in data['response']

class TestQueryHandling:
    """Test query processing logic"""
    
    def test_basketball_queries(self, client):
        """Test various basketball-related queries"""
        queries = [
            "Who is the leading scorer?",
            "Show me Lauren Betts stats",
            "Top rebounders this season",
            "Team statistics"
        ]
        
        for query in queries:
            response = client.post('/query',
                                 data=json.dumps({'query': query}),
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['response']) > 0
            assert data['tokens'] > 0

class TestSessions:
    """Test session management"""
    
    def test_chat_history(self, client):
        """Test chat history functionality"""
        # Make a query
        client.post('/query',
                   data=json.dumps({'query': 'test query'}),
                   content_type='application/json')
        
        # Check history
        response = client.get('/history')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_clear_chat(self, client):
        """Test clearing chat history"""
        # Make a query first
        client.post('/query',
                   data=json.dumps({'query': 'test'}),
                   content_type='application/json')
        
        # Clear chat
        response = client.post('/clear-chat')
        assert response.status_code == 200
        
        # Check history is empty
        response = client.get('/history')
        data = json.loads(response.data)
        assert len(data) == 0

if __name__ == '__main__':
    pytest.main([__file__]) 