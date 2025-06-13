import os
import sys
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM

# Add the parent directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# No need for a custom LLM class as we'll use LangChain's ChatAnthropic implementation

class LLMManager:
    """Manages LLM model initialization and usage."""
    
    def __init__(self, model_name="claude-3-5-sonnet-20241022", embedding_model="sentence-transformers/all-mpnet-base-v2"):
        """Initialize LLM and embedding models.
        
        Args:
            model_name: Name of the Anthropic model to use (default: claude-3-haiku-20240307)
            embedding_model: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.llm = None
        self.embeddings = None
        self.initialization_error = None
        
        # Initialize models
        self._initialize_llm()
        self._initialize_embeddings()
    
    def _initialize_llm(self):
        """Initialize the Anthropic Claude model."""
        try:
            # Get API token from environment variable
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                error_msg = (
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Please set your Anthropic API key in the .env file or environment variables. "
                    "You can get an API key at https://console.anthropic.com/"
                )
                self.initialization_error = error_msg
                raise ValueError(error_msg)
                
            self.llm = ChatAnthropic(
                api_key=anthropic_api_key,
                model_name=self.model_name,
                temperature=0.7,
                max_tokens=1000
            )
            self.initialization_error = None
        except Exception as e:
            error_msg = f"Error initializing Anthropic LLM: {str(e)}"
            self.initialization_error = error_msg
            print(error_msg)
            
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            # Using HuggingFace embeddings as Anthropic doesn't provide embedding models
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        except Exception as e:
            error_msg = f"Error initializing embeddings: {str(e)}"
            print(error_msg)
            # Don't set initialization_error here as it's not critical for basic functionality
            
    def generate_text(self, prompt, max_tokens=1000):
        """Generate text with the LLM."""
        if not self.llm:
            if self.initialization_error:
                raise ValueError(f"LLM not initialized: {self.initialization_error}")
            else:
                raise ValueError("LLM not initialized: Unknown error during initialization")
        
        return self.llm.invoke(prompt).content
    
    def get_embeddings(self, texts):
        """Get embeddings for the given texts."""
        if not self.embeddings:
            raise ValueError("Embeddings model not initialized")
        
        return self.embeddings.embed_documents(texts)