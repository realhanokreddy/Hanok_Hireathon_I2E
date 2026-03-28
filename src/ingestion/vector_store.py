"""
Vector store module for embedding and indexing document chunks.
Supports FAISS for efficient similarity search with metadata filtering.
Supports both local (sentence-transformers) and OpenAI embeddings.
"""
import logging
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logging.warning("FAISS not installed. Install with: pip install faiss-cpu")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.config import get_config
from src.ingestion.chunker import DocumentChunk


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from vector store."""
    chunk_id: str
    content: str
    score: float
    section_number: str
    section_title: str
    page_start: int
    page_end: int
    metadata: Dict[str, Any]
    chunk_type: str
    

class VectorStore:
    """Vector store for document chunks using FAISS."""
    
    def __init__(self, config=None):
        """
        Initialize vector store.
        
        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self.vector_config = self.config.get_vector_store_config()
        self.embedding_config = self.config.get('embedding', {})
        self.openai_config = self.config.get_openai_config()
        
        self.dimension = self.vector_config.get('dimension', 384)
        self.index_type = self.vector_config.get('index_type', 'flat')
        self.distance_metric = self.vector_config.get('distance_metric', 'cosine')
        
        # Determine embedding provider
        self.embedding_provider = self.embedding_config.get('provider', 'local')
        
        # Initialize embedding model
        if self.embedding_provider == 'local':
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
            model_name = self.embedding_config.get('local_model', 'sentence-transformers/all-MiniLM-L6-v2')
            logger.info(f"Loading local embedding model: {model_name}")
            self.local_model = SentenceTransformer(model_name)
            self.client = None
            self.embedding_model = None
        elif self.embedding_provider == 'openai':
            if not HAS_OPENAI:
                raise ImportError("openai not installed. Install with: pip install openai")
            api_key = self.openai_config.get('api_key')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY.")
            self.client = OpenAI(api_key=api_key)
            self.embedding_model = self.openai_config.get('embedding_model', 'text-embedding-3-large')
            self.local_model = None
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
        
        # Vector storage
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        logger.info(f"Initialized vector store: provider={self.embedding_provider}, dimension={self.dimension}, index_type={self.index_type}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using configured provider.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            if self.embedding_provider == 'local':
                embedding = self.local_model.encode(text, convert_to_numpy=True)
                return embedding.astype(np.float32)
            elif self.embedding_provider == 'openai':
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts
            batch_size: Batch size for API calls
            
        Returns:
            Array of embeddings
        """
        if self.embedding_provider == 'local':
            # Sentence transformers handles batching automatically
            logger.info(f"Generating {len(texts)} embeddings with local model...")
            embeddings = self.local_model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
        elif self.embedding_provider == 'openai':
            from tqdm import tqdm
            
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
                batch = texts[i:i + batch_size]
                
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.embedding_model
                    )
                    
                    batch_embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    logger.error(f"Error embedding batch {i}: {e}")
                    # Add zero vectors for failed embeddings
                    embeddings.extend([np.zeros(self.dimension, dtype=np.float32)] * len(batch))
            
            return np.array(embeddings)
    
    def build_index(self, chunks: List[DocumentChunk]):
        """
        Build vector index from chunks.
        
        Args:
            chunks: List of document chunks
        """
        if not HAS_FAISS:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        logger.info(f"Building vector index for {len(chunks)} chunks")
        
        self.chunks = chunks
        
        # Get texts to embed with context
        texts = [chunk.get_context_string() for chunk in chunks]
        
        # Get embeddings
        logger.info("Getting embeddings from OpenAI...")
        embeddings = self.get_embeddings_batch(texts)
        self.embeddings = embeddings
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        
        if self.index_type == 'flat':
            # Flat index for exact search
            if self.distance_metric == 'cosine':
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
            else:
                self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance
        
        elif self.index_type == 'ivf':
            # IVF index for faster approximate search
            nlist = min(100, len(chunks) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train the index
            self.index.train(embeddings)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add vectors to index
        self.index.add(embeddings)
        
        logger.info(f"Vector index built with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Get query embedding
        query_embedding = self.get_embedding(query).reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.distance_metric == 'cosine':
            faiss.normalize_L2(query_embedding)
        
        # Search
        # Get more results if we need to filter
        search_k = top_k * 3 if filters else top_k
        distances, indices = self.index.search(query_embedding, min(search_k, len(self.chunks)))
        
        # Convert to search results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Apply filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            
            # Convert distance to score (higher is better)
            if self.distance_metric == 'cosine':
                score = float(dist)  # Already similarity for inner product
            else:
                score = 1.0 / (1.0 + float(dist))  # Convert L2 distance to similarity
            
            result = SearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=score,
                section_number=chunk.section_number,
                section_title=chunk.section_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                metadata=chunk.metadata,
                chunk_type=chunk.chunk_type
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk matches metadata filters."""
        for key, value in filters.items():
            if key == 'chunk_type' and chunk.chunk_type != value:
                return False
            elif key == 'section_number' and chunk.section_number != value:
                return False
            elif key == 'min_page' and chunk.page_start < value:
                return False
            elif key == 'max_page' and chunk.page_end > value:
                return False
            elif key in chunk.metadata and chunk.metadata[key] != value:
                return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Document chunk or None
        """
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_section(self, section_number: str) -> List[DocumentChunk]:
        """
        Get all chunks for a section.
        
        Args:
            section_number: Section number
            
        Returns:
            List of chunks
        """
        return [
            chunk for chunk in self.chunks
            if chunk.section_number == section_number
        ]
    
    def save(self, path: str):
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index:
            index_path = path / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")
        
        # Save chunks
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved chunks to {chunks_path}")
        
        # Save embeddings
        if self.embeddings is not None:
            embeddings_path = path / "embeddings.npy"
            np.save(embeddings_path, self.embeddings)
            logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save configuration
        config_path = path / "vector_store_config.json"
        config_data = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'distance_metric': self.distance_metric,
            'embedding_model': self.embedding_model,
            'num_chunks': len(self.chunks)
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Saved config to {config_path}")
    
    def load(self, path: str):
        """
        Load vector store from disk.
        
        Args:
            path: Directory path to load from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        # Load FAISS index
        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load chunks
        chunks_path = path / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
        
        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings from {embeddings_path}")
        
        # Load config
        config_path = path / "vector_store_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Loaded vector store config: {config_data}")


def main():
    """Main function for building vector store."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build vector store from chunks')
    parser.add_argument('--build', action='store_true', help='Build new vector store')
    parser.add_argument('--load', action='store_true', help='Load existing vector store')
    parser.add_argument('--query', type=str, help='Query to test')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = get_config()
    vector_store = VectorStore(config)
    
    if args.build:
        # Load chunks
        chunks_path = config.get('paths.chunks_output')
        logger.info(f"Loading chunks from {chunks_path}")
        
        with open(chunks_path, 'r') as f:
            chunks_data = json.load(f)
        
        chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
        
        # Build index
        vector_store.build_index(chunks)
        
        # Save
        save_path = config.get('paths.vector_store')
        vector_store.save(save_path)
        
        print(f"\n✓ Vector store built!")
        print(f"  Chunks indexed: {len(chunks)}")
        print(f"  Saved to: {save_path}")
    
    elif args.load:
        # Load existing vector store
        load_path = config.get('paths.vector_store')
        vector_store.load(load_path)
        
        print(f"\n✓ Vector store loaded!")
        print(f"  Chunks: {len(vector_store.chunks)}")
    
    if args.query:
        # Test query
        if vector_store.index is None:
            load_path = config.get('paths.vector_store')
            vector_store.load(load_path)
        
        logger.info(f"Searching for: {args.query}")
        results = vector_store.search(args.query, top_k=5)
        
        print(f"\n🔍 Search results for: '{args.query}'\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Section {result.section_number}: {result.section_title}")
            print(f"   Score: {result.score:.3f} | Page: {result.page_start}")
            print(f"   {result.content[:200]}...")
            print()


if __name__ == '__main__':
    main()
