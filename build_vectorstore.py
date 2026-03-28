"""
Quick script to build vector store from existing chunks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.ingestion.vector_store import VectorStore
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

print("=" * 80)
print("BUILDING VECTOR STORE")
print("=" * 80)
print()

try:
    # Load config
    config = get_config()
    
    # Load chunks
    chunks_path = config.get('paths.chunks_output')
    print(f"Loading chunks from: {chunks_path}")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    print(f"✓ Loaded {len(chunks_data)} chunks")
    print()
    
    # Initialize vector store
    print("Initializing vector store...")
    vector_store = VectorStore(config)
    
    # Add chunks
    print("Adding chunks and building index...")
    from src.ingestion.chunker import DocumentChunk
    
    chunks = [DocumentChunk.from_dict(c) for c in chunks_data]
    vector_store.add_chunks(chunks)
    
    # Build index
    print("Building FAISS index...")
    vector_store.build_index()
    
    # Save
    store_path = config.get('paths.vector_store')
    print(f"Saving to: {store_path}")
    vector_store.save(store_path)
    
    print()
    print("=" * 80)
    print("✅ VECTOR STORE BUILT SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("You can now run the Streamlit UI:")
    print("  .\\venv\\Scripts\\streamlit run app.py")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
