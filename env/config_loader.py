"""
Enhanced environment configuration loader for NASA QA System.
Loads configuration from env/.env file with proper validation and type conversion.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dotenv import load_dotenv


class EnvConfig:
    """Enhanced environment configuration loader."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize environment configuration.
        
        Args:
            env_file: Path to .env file. Defaults to env/.env
        """
        # Determine project root (parent of src)
        self.project_root = Path(__file__).parent.parent
        
        # Load env file
        if env_file:
            env_path = Path(env_file)
        else:
            # Try env/.env first, then root .env
            env_path = self.project_root / "env" / ".env"
            if not env_path.exists():
                env_path = self.project_root / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded environment from: {env_path}")
        else:
            print(f"⚠ Warning: No .env file found at {env_path}")
            print(f"  Copy env/.env.example to env/.env and configure your settings.")
    
    def get(self, key: str, default: Any = None, cast: type = str) -> Any:
        """
        Get environment variable with type casting.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            cast: Type to cast to (str, int, float, bool)
            
        Returns:
            Value from environment or default
        """
        value = os.getenv(key, default)
        
        if value is None:
            return default
        
        # Type casting
        if cast == bool:
            return str(value).lower() in ('true', '1', 'yes', 'on')
        elif cast == int:
            return int(value)
        elif cast == float:
            return float(value)
        else:
            return str(value)
    
    # =============================================================================
    # LLM Provider Configuration
    # =============================================================================
    
    @property
    def llm_provider(self) -> str:
        """Get LLM provider: 'groq' or 'openai'"""
        return self.get('LLM_PROVIDER', 'groq')
    
    # =============================================================================
    # Groq Configuration
    # =============================================================================
    
    @property
    def groq_api_key(self) -> Optional[str]:
        """Get Groq API key"""
        key = self.get('GROQ_API_KEY', '')
        return key if key and key != 'your_groq_api_key_here' else None
    
    @property
    def groq_model(self) -> str:
        """Get Groq model name"""
        return self.get('GROQ_MODEL', 'llama3-70b-8192')
    
    @property
    def groq_temperature(self) -> float:
        """Get Groq temperature"""
        return self.get('GROQ_TEMPERATURE', 0.1, float)
    
    @property
    def groq_max_tokens(self) -> int:
        """Get Groq max tokens"""
        return self.get('GROQ_MAX_TOKENS', 2000, int)
    
    # =============================================================================
    # OpenAI Configuration
    # =============================================================================
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key"""
        key = self.get('OPENAI_API_KEY', '')
        return key if key and key != 'your_openai_api_key_here' else None
    
    @property
    def openai_model(self) -> str:
        """Get OpenAI model name"""
        return self.get('OPENAI_MODEL', 'gpt-4-turbo-preview')
    
    @property
    def openai_temperature(self) -> float:
        """Get OpenAI temperature"""
        return self.get('OPENAI_TEMPERATURE', 0.1, float)
    
    @property
    def openai_max_tokens(self) -> int:
        """Get OpenAI max tokens"""
        return self.get('OPENAI_MAX_TOKENS', 2000, int)
    
    # =============================================================================
    # Embedding Configuration
    # =============================================================================
    
    @property
    def embedding_provider(self) -> str:
        """Get embedding provider: 'local' or 'openai'"""
        return self.get('EMBEDDING_PROVIDER', 'local')
    
    @property
    def local_embedding_model(self) -> str:
        """Get local embedding model name"""
        return self.get('LOCAL_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    
    @property
    def local_embedding_dimension(self) -> int:
        """Get local embedding dimension"""
        return self.get('LOCAL_EMBEDDING_DIMENSION', 384, int)
    
    @property
    def openai_embedding_model(self) -> str:
        """Get OpenAI embedding model name"""
        return self.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
    
    @property
    def openai_embedding_dimension(self) -> int:
        """Get OpenAI embedding dimension"""
        return self.get('OPENAI_EMBEDDING_DIMENSION', 3072, int)
    
    # =============================================================================
    # PDF Parser Configuration
    # =============================================================================
    
    @property
    def pdf_primary_parser(self) -> str:
        """Get primary PDF parser"""
        return self.get('PDF_PRIMARY_PARSER', 'docling')
    
    @property
    def pdf_fallback_parser(self) -> str:
        """Get fallback PDF parser"""
        return self.get('PDF_FALLBACK_PARSER', 'pymupdf')
    
    @property
    def pdf_enable_ocr(self) -> bool:
        """Check if OCR is enabled"""
        return self.get('PDF_ENABLE_OCR', True, bool)
    
    @property
    def pdf_extract_images(self) -> bool:
        """Check if image extraction is enabled"""
        return self.get('PDF_EXTRACT_IMAGES', True, bool)
    
    @property
    def pdf_extract_tables(self) -> bool:
        """Check if table extraction is enabled"""
        return self.get('PDF_EXTRACT_TABLES', True, bool)
    
    # =============================================================================
    # Chunking Configuration
    # =============================================================================
    
    @property
    def chunk_strategy(self) -> str:
        """Get chunking strategy"""
        return self.get('CHUNK_STRATEGY', 'hierarchical')
    
    @property
    def chunk_size(self) -> int:
        """Get chunk size"""
        return self.get('CHUNK_SIZE', 800, int)
    
    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap"""
        return self.get('CHUNK_OVERLAP', 150, int)
    
    @property
    def chunk_respect_boundaries(self) -> bool:
        """Check if respecting section boundaries"""
        return self.get('CHUNK_RESPECT_BOUNDARIES', True, bool)
    
    @property
    def chunk_include_parent_context(self) -> bool:
        """Check if including parent context"""
        return self.get('CHUNK_INCLUDE_PARENT_CONTEXT', True, bool)
    
    @property
    def chunk_min_size(self) -> int:
        """Get minimum chunk size"""
        return self.get('CHUNK_MIN_SIZE', 200, int)
    
    # =============================================================================
    # Metadata Configuration
    # =============================================================================
    
    @property
    def metadata_extract_acronyms(self) -> bool:
        """Check if extracting acronyms"""
        return self.get('METADATA_EXTRACT_ACRONYMS', True, bool)
    
    @property
    def metadata_resolve_cross_refs(self) -> bool:
        """Check if resolving cross-references"""
        return self.get('METADATA_RESOLVE_CROSS_REFS', True, bool)
    
    @property
    def metadata_extract_hierarchy(self) -> bool:
        """Check if extracting hierarchy"""
        return self.get('METADATA_EXTRACT_HIERARCHY', True, bool)
    
    # =============================================================================
    # Vector Store Configuration
    # =============================================================================
    
    @property
    def vector_store_type(self) -> str:
        """Get vector store type"""
        return self.get('VECTOR_STORE_TYPE', 'faiss')
    
    @property
    def vector_store_index_type(self) -> str:
        """Get vector store index type"""
        return self.get('VECTOR_STORE_INDEX_TYPE', 'flat')
    
    @property
    def vector_store_distance_metric(self) -> str:
        """Get vector store distance metric"""
        return self.get('VECTOR_STORE_DISTANCE_METRIC', 'cosine')
    
    @property
    def vector_store_dimension(self) -> int:
        """Get vector store dimension"""
        return self.get('VECTOR_STORE_DIMENSION', 384, int)
    
    # =============================================================================
    # Retrieval Configuration
    # =============================================================================
    
    @property
    def retrieval_strategy(self) -> str:
        """Get retrieval strategy"""
        return self.get('RETRIEVAL_STRATEGY', 'hybrid')
    
    @property
    def retrieval_top_k(self) -> int:
        """Get retrieval top k"""
        return self.get('RETRIEVAL_TOP_K', 5, int)
    
    @property
    def retrieval_enable_multi_hop(self) -> bool:
        """Check if multi-hop is enabled"""
        return self.get('RETRIEVAL_ENABLE_MULTI_HOP', True, bool)
    
    @property
    def retrieval_max_hops(self) -> int:
        """Get max hops for multi-hop retrieval"""
        return self.get('RETRIEVAL_MAX_HOPS', 2, int)
    
    @property
    def retrieval_vector_weight(self) -> float:
        """Get vector weight for hybrid retrieval"""
        return self.get('RETRIEVAL_VECTOR_WEIGHT', 0.7, float)
    
    @property
    def retrieval_keyword_weight(self) -> float:
        """Get keyword weight for hybrid retrieval"""
        return self.get('RETRIEVAL_KEYWORD_WEIGHT', 0.3, float)
    
    # =============================================================================
    # Generation Configuration
    # =============================================================================
    
    @property
    def generation_include_citations(self) -> bool:
        """Check if including citations"""
        return self.get('GENERATION_INCLUDE_CITATIONS', True, bool)
    
    @property
    def generation_citation_format(self) -> str:
        """Get citation format"""
        return self.get('GENERATION_CITATION_FORMAT', 'section_page')
    
    @property
    def generation_min_confidence(self) -> float:
        """Get minimum confidence threshold"""
        return self.get('GENERATION_MIN_CONFIDENCE', 0.3, float)
    
    # =============================================================================
    # Path Configuration
    # =============================================================================
    
    @property
    def data_dir(self) -> str:
        """Get data directory"""
        return self.get('DATA_DIR', './data')
    
    @property
    def pdf_file(self) -> str:
        """Get PDF file path"""
        return self.get('PDF_FILE', './data/nasa_handbook.pdf')
    
    @property
    def parsed_output(self) -> str:
        """Get parsed output path"""
        return self.get('PARSED_OUTPUT', './data/parsed/document.json')
    
    @property
    def chunks_output(self) -> str:
        """Get chunks output path"""
        return self.get('CHUNKS_OUTPUT', './data/chunks/chunks.json')
    
    @property
    def vector_store_path(self) -> str:
        """Get vector store path"""
        return self.get('VECTOR_STORE_PATH', './data/vectorstore')
    
    @property
    def logs_dir(self) -> str:
        """Get logs directory"""
        return self.get('LOGS_DIR', './logs')
    
    # =============================================================================
    # Download Configuration
    # =============================================================================
    
    @property
    def nasa_handbook_url(self) -> str:
        """Get NASA handbook URL"""
        return self.get('NASA_HANDBOOK_URL', 
                       'https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf')
    
    @property
    def download_timeout(self) -> int:
        """Get download timeout"""
        return self.get('DOWNLOAD_TIMEOUT', 300, int)
    
    # =============================================================================
    # Logging Configuration
    # =============================================================================
    
    @property
    def log_level(self) -> str:
        """Get log level"""
        return self.get('LOG_LEVEL', 'INFO')
    
    @property
    def log_file(self) -> str:
        """Get log file path"""
        return self.get('LOG_FILE', './logs/qa_system.log')
    
    # =============================================================================
    # System Configuration
    # =============================================================================
    
    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get('DEBUG_MODE', False, bool)
    
    @property
    def enable_cache(self) -> bool:
        """Check if cache is enabled"""
        return self.get('ENABLE_CACHE', True, bool)
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory"""
        return self.get('CACHE_DIR', './data/cache')


# Global instance
_env_config = None


def get_env_config() -> EnvConfig:
    """Get global environment configuration instance."""
    global _env_config
    if _env_config is None:
        _env_config = EnvConfig()
    return _env_config


if __name__ == '__main__':
    # Test configuration loading
    config = get_env_config()
    print("\n=== Configuration Summary ===")
    print(f"LLM Provider: {config.llm_provider}")
    print(f"Embedding Provider: {config.embedding_provider}")
    print(f"Groq Model: {config.groq_model}")
    print(f"Vector Store Dimension: {config.vector_store_dimension}")
    print(f"Chunk Size: {config.chunk_size}")
    print(f"Retrieval Top K: {config.retrieval_top_k}")
    print(f"Multi-hop Enabled: {config.retrieval_enable_multi_hop}")
    print("=" * 30)
