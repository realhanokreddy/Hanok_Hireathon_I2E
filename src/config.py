"""
Configuration loader for the NASA Technical Manual QA System.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class Config:
    """Configuration management for the QA system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file and environment variables.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Load environment variables from .env file if it exists
        # Try both root .env and env/.env
        load_dotenv()  # Load from current dir
        env_path = Path(__file__).parent.parent / 'env' / '.env'
        if env_path.exists():
            load_dotenv(env_path)  # Also load from env/.env
        
        # Load configuration from YAML
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_env_overrides(self):
        """Override configuration with environment variables."""
        # OpenAI API key from environment
        if os.getenv('OPENAI_API_KEY'):
            self._config['openai']['api_key'] = os.getenv('OPENAI_API_KEY')
        
        # Groq API key from environment
        if os.getenv('GROQ_API_KEY'):
            if 'groq' not in self._config:
                self._config['groq'] = {}
            self._config['groq']['api_key'] = os.getenv('GROQ_API_KEY')
        
        # Gemini API key from environment
        if os.getenv('GEMINI_API_KEY'):
            if 'gemini' not in self._config:
                self._config['gemini'] = {}
            self._config['gemini']['api_key'] = os.getenv('GEMINI_API_KEY')
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.get('paths.data_dir'),
            os.path.dirname(self.get('paths.parsed_output')),
            os.path.dirname(self.get('paths.chunks_output')),
            self.get('paths.vector_store'),
            self.get('paths.logs'),
            self.get('performance.cache_dir'),
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'openai.api_key')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return self._config.get('openai', {})
    
    def get_pdf_parser_config(self) -> Dict[str, Any]:
        """Get PDF parser configuration."""
        return self._config.get('pdf_parser', {})
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration."""
        return self._config.get('chunking', {})
    
    def get_metadata_config(self) -> Dict[str, Any]:
        """Get metadata extraction configuration."""
        return self._config.get('metadata', {})
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return self._config.get('vector_store', {})
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return self._config.get('retrieval', {})
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get answer generation configuration."""
        return self._config.get('generation', {})
    
    def get_paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self._config.get('paths', {})


# Global configuration instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get or create the global configuration instance.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
