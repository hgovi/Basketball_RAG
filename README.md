# UCLA Women's Basketball RAG Analytics 🏀

> An intelligent Retrieval-Augmented Generation (RAG) chatbot for UCLA Women's Basketball statistics and analytics, powered by Claude 3.5 Sonnet.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude%203.5-orange.svg)](https://www.anthropic.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Overview

This application combines advanced Natural Language Processing with sports analytics to provide an intelligent chatbot that answers complex questions about UCLA Women's Basketball statistics. Built with Claude 3.5 Sonnet and a sophisticated RAG pipeline, it processes natural language queries and returns comprehensive basketball insights through entity extraction, SQL query generation, and intelligent response formatting.

### Key Features

- **🧠 Advanced RAG Pipeline**: Entity extraction → SQL generation → Query validation → Execution → Natural language response
- **🔍 Intelligent Query Processing**: Handles complex basketball questions with contextual understanding
- **📊 Comprehensive Analytics**: Player statistics, team performance, game-by-game analysis, and comparative insights
- **⚡ Real-time Performance**: Fast query processing with comprehensive error handling and fallback strategies
- **🎨 Modern UI**: Beautiful, responsive web interface with real-time chat functionality
- **🔒 Production Ready**: Thread-safe database connections, proper logging, and health monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Anthropic API key (Claude 3.5 Sonnet)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ucla-basketball-rag.git
   cd ucla-basketball-rag
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   export FLASK_SECRET_KEY="your-secret-key"  # Optional
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5001`

## 🏗️ Project Architecture

```
ucla-basketball-rag/
├── app.py                    # Flask web application
├── src/                      # Core RAG pipeline components
│   ├── rag_pipeline.py       # Main RAG orchestration
│   ├── entity_extractor.py   # NLP entity extraction
│   ├── query_generator.py    # SQL query generation
│   ├── db_connector.py       # Database operations
│   └── llm_utils.py          # LLM integration utilities
├── data/                     # Database and datasets
│   ├── ucla_wbb.db           # SQLite database (402 records)
│   └── uclawbb_season.csv    # Raw CSV data
├── templates/                # HTML templates
│   └── index.html            # Main chat interface
├── static/                   # Frontend assets
│   └── css/style.css         # Modern styling
├── tests/                    # Test suite
│   └── test_app.py           # Application tests
├── logs/                     # Application logs (auto-created)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🎯 RAG Pipeline Flow

The system uses an intelligent multi-step process to understand and answer basketball queries:

```
User Query → Entity Extraction → SQL Generation → Validation → Execution → Response Generation
```

### Detailed Process:

1. **🎤 Query Reception**: User submits natural language basketball question
2. **🔍 Entity Extraction**: AI identifies players, statistics, opponents, dates, and game context
3. **⚙️ SQL Generation**: Converts entities and intent into optimized SQLite queries
4. **✅ Query Validation**: Ensures SQL safety, syntax correctness, and data availability
5. **💾 Database Execution**: Runs validated query against UCLA basketball database
6. **🔄 Fallback Handling**: Multiple fallback strategies for query failures or empty results
7. **📊 Result Processing**: Formats and structures query results for optimal presentation
8. **💬 Response Generation**: Creates natural language responses using Claude 3.5 Sonnet

### Error Handling & Fallbacks

- **Simplified Aggregation**: Converts complex queries to basic aggregations
- **Basic Select Conversion**: Falls back to simple SELECT statements
- **Player Lookup**: Direct player name searches when complex queries fail
- **Empty Result Handling**: Suggests alternative queries when no data is found

## 📊 Supported Query Types

### Player Statistics
```
"How many points does Lauren Betts average?"
"Show me Kiki Rice's assist numbers this season"
"Who is UCLA's leading scorer?"
"What are Londynn Jones' shooting percentages?"
```

### Game Analysis
```
"Who scored the most points against USC?"
"How many three pointers did Kiki Rice make against Richmond?"
"Which player had the most rebounds in the LSU game?"
"Show me stats from the Oregon game"
```

### Team Performance
```
"What are UCLA's team totals this season?"
"How many games has UCLA played?"
"Show me the team's shooting percentages"
"What was our highest scoring game?"
```

### Comparative Analysis
```
"Compare Lauren Betts and Londynn Jones rebounding"
"Top 5 scorers this season"
"Who are the best three-point shooters?"
"Show me assist leaders and their averages"
```

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Development Mode
```bash
export FLASK_ENV=development
python app.py
```

## 🚀 Deployment

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

### Environment Variables
```bash
ANTHROPIC_API_KEY=your-anthropic-api-key
FLASK_SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
```

### Health Monitoring
The application includes built-in health monitoring endpoints:
- **Health Check**: `GET /health` - Database connectivity and system status
- **Statistics**: `GET /stats` - Usage metrics and database statistics

## 📈 Performance Metrics

- **Response Time**: < 2 seconds average query processing
- **Database**: 402 player-game statistical records
- **Accuracy**: Handles complex natural language queries with high precision
- **Reliability**: Comprehensive error handling with multiple fallback strategies
- **Concurrency**: Thread-safe SQLite connections for production use

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Follow coding standards** (PEP 8, proper docstrings)
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add comprehensive docstrings to all functions
- Include unit tests for new features
- Ensure thread safety for database operations
- Update README for significant changes

## 📝 API Reference

### Core Endpoints

#### Query Processing
```http
POST /query
Content-Type: application/json

{
  "query": "Who is the top scorer this season?"
}

Response:
{
  "response": "Based on the UCLA women's basketball statistics...",
  "tokens": 45,
  "total_tokens": 150
}
```

#### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "database": "connected",
  "records": 402,
  "version": "1.0.0"
}
```

#### Statistics
```http
GET /stats

Response:
{
  "total_tokens": 1500,
  "chat_sessions": 12,
  "games_in_db": 15,
  "players_tracked": 12,
  "rag_status": "active"
}
```

#### Chat History
```http
GET /history

Response: [
  {
    "timestamp": "2024-01-15T10:30:00",
    "query": "Who scored the most points?",
    "response": "Lauren Betts leads the team...",
    "tokens": 32
  }
]
```

## 🔧 Technical Details

### Database Schema
The SQLite database contains UCLA women's basketball statistics with the following key fields:
- **Player Information**: Name, position, class
- **Game Details**: Date, opponent, home/away
- **Statistics**: Points, rebounds, assists, shooting percentages, etc.

### RAG Components
- **Entity Extractor**: Identifies basketball-specific entities (players, stats, games)
- **Query Generator**: Converts natural language to SQL with basketball domain knowledge
- **Database Connector**: Thread-safe SQLite operations with connection pooling
- **LLM Manager**: Anthropic Claude 3.5 Sonnet integration for NLP tasks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCLA Women's Basketball** for the inspiration and data
- **Anthropic** for Claude 3.5 Sonnet LLM capabilities
- **Flask** for the web framework
- **SQLite** for efficient data storage

## Pushing to a GitHub Repo (Removing API Key)

Before pushing your code to a GitHub repo, please remove your API key (and any other sensitive information) so that it isn't accidentally committed. You can do this by:

1. Renaming (or deleting) your local .env file (for example, rename it to .env.local) so that your ANTHROPIC_API_KEY isn't pushed.
2. (Optional) Create a .env.example (or a similar template) file (with dummy values) so that other developers know which environment variables are needed.
3. Ensure that your .gitignore (already updated) ignores the .env file (and __pycache__ and logs) so that sensitive data isn't pushed.

Once you've removed your API key, you can safely push your code to your GitHub repo.

---

**Built with ❤️ for UCLA Women's Basketball Analytics** 