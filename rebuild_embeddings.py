"""
Rebuild embeddings from existing chunks.pkl without re-parsing the PDF.
Run this after any change to DocumentChunk.get_context_string() to update
the FAISS index without going through the full ingestion pipeline.

Usage:
    python rebuild_embeddings.py
"""
import sys
import pickle
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = Path("data/vectorstore")


def main():
    # ── Load chunks ────────────────────────────────────────────────────────────
    chunks_pkl = VECTORSTORE_DIR / "chunks.pkl"
    if not chunks_pkl.exists():
        logger.error(f"chunks.pkl not found at {chunks_pkl}. Run ingestion first.")
        sys.exit(1)

    with open(chunks_pkl, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_pkl}")

    # ── Build context strings using UPDATED get_context_string() ───────────────
    logger.info("Building context strings ...")
    texts = [chunk.get_context_string() for chunk in chunks]

    # Spot-check: show the table of interest
    for c in chunks:
        if c.chunk_id == "chunk_520":
            logger.info(f"\nchunk_520 context string (first 300 chars):\n{texts[chunks.index(c)][:300]}")
            break

    # ── Embed ─────────────────────────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed.")
        sys.exit(1)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Embedding {len(texts)} chunks ...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # ── Rebuild FAISS index ────────────────────────────────────────────────────
    try:
        import faiss
    except ImportError:
        logger.error("faiss-cpu not installed.")
        sys.exit(1)

    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index (cosine, dim={dim}) ...")
    normed = embeddings.copy()
    faiss.normalize_L2(normed)
    index = faiss.IndexFlatIP(dim)
    index.add(normed)
    logger.info(f"Index has {index.ntotal} vectors")

    # ── Save ───────────────────────────────────────────────────────────────────
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(VECTORSTORE_DIR / "index.faiss"))
    logger.info("Saved index.faiss")

    np.save(VECTORSTORE_DIR / "embeddings.npy", embeddings)
    logger.info("Saved embeddings.npy")

    # ── Verify chunk_520 ──────────────────────────────────────────────────────
    logger.info("\n── Verification ──────────────────────────────────────────────")
    idx520 = next((i for i, c in enumerate(chunks) if c.chunk_id == "chunk_520"), None)
    if idx520 is not None:
        query = "Purpose and Results for Life-Cycle Reviews for Spaceflight Projects"
        qvec = model.encode([query], normalize_embeddings=True).astype(np.float32)
        sims = normed @ qvec[0]
        rank = int((sims > sims[idx520]).sum()) + 1
        logger.info(f"chunk_520 similarity: {sims[idx520]:.4f}  |  rank: {rank} / {len(chunks)}")

        top5_idx = np.argsort(sims)[::-1][:5]
        logger.info("Top-5 results:")
        for r, i in enumerate(top5_idx):
            c = chunks[i]
            snippet = c.content[:80].replace("\n", " ")
            logger.info(f"  #{r+1} sim={sims[i]:.4f} | {c.chunk_id} | sec={c.section_number} | {snippet}")
    else:
        logger.warning("chunk_520 not found in chunks list")

    logger.info("\nDone. The retriever will use the updated embeddings on next startup.")


if __name__ == "__main__":
    main()
