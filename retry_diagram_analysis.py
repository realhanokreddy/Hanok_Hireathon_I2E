"""
retry_diagram_analysis.py

Post-ingestion script that retries Gemini image analysis for diagram chunks
that were skipped during ingestion due to rate limiting.

Run this after your Gemini API quota has reset:
    python retry_diagram_analysis.py

What it does:
  1. Loads the existing FAISS index + chunks from ./data/vectorstore/
  2. Finds all diagram_ref chunks where has_relationship_analysis=False
  3. Re-extracts the original image bytes from the PDF
  4. Calls Gemini to get description + extracted text
  5. Updates the chunk content in-place
  6. Re-embeds the updated chunks
  7. Rebuilds and saves the FAISS index with the new embeddings

The ingestion process is never touched — this script is fully standalone.
"""
import sys
import logging
import pickle
import re
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_diagram_id(diagram_id: str):
    """
    Parse a diagram_id like 'diagram_p6_i1' into (page_1indexed, img_1indexed).
    Returns None if the format is not recognised (e.g. fallback 'diagram_0' ids).
    """
    m = re.match(r'diagram_p(\d+)_i(\d+)', diagram_id or '')
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _extract_image_bytes(pdf_path: str, page_1indexed: int, img_1indexed: int):
    """
    Re-extract raw image bytes from the PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.
        page_1indexed: 1-based page number (as stored in diagram metadata).
        img_1indexed: 1-based image index on that page.

    Returns:
        Raw image bytes, or None if extraction fails.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
        return None

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_1indexed - 1]          # fitz uses 0-based indexing
        images = page.get_images(full=True)

        if img_1indexed - 1 >= len(images):
            logger.warning(
                f"Image index {img_1indexed} out of range on page {page_1indexed} "
                f"({len(images)} image(s) found on that page). Skipping."
            )
            return None

        img_info = images[img_1indexed - 1]
        xref = img_info[0]
        image_dict = doc.extract_image(xref)
        return image_dict['image']

    except Exception as e:
        logger.warning(
            f"Could not extract image from page {page_1indexed}, "
            f"image {img_1indexed}: {e}"
        )
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    try:
        import faiss
    except ImportError:
        logger.error("FAISS not installed. Install with: pip install faiss-cpu")
        sys.exit(1)

    from src.config import get_config
    from src.ingestion.vector_store import VectorStore
    from src.ingestion.pdf_parser import PDFParser

    config = get_config()
    store_path = Path(config.get('paths.vector_store', './data/vectorstore'))
    pdf_path = config.get('paths.pdf_file', './data/nasa_handbook.pdf')

    # ── 1. Load existing vector store ──────────────────────────────────────
    chunks_file = store_path / 'chunks.pkl'
    embeddings_file = store_path / 'embeddings.npy'
    index_file = store_path / 'index.faiss'

    if not chunks_file.exists():
        logger.error(f"chunks.pkl not found at {store_path}. Run ingestion first.")
        sys.exit(1)
    if not embeddings_file.exists():
        logger.error(f"embeddings.npy not found at {store_path}. Run ingestion first.")
        sys.exit(1)

    logger.info(f"Loading vector store from {store_path} ...")
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)  # List[DocumentChunk], order matches FAISS

    embeddings = np.load(embeddings_file)  # shape (N, dim)
    logger.info(f"Loaded {len(chunks)} chunks — embeddings shape: {embeddings.shape}")

    # ── 2. Find caption-only diagram chunks ────────────────────────────────
    pending = [
        (idx, chunk)
        for idx, chunk in enumerate(chunks)
        if chunk.chunk_type == 'diagram_ref'
        and not chunk.metadata.get('has_relationship_analysis', False)
    ]

    if not pending:
        logger.info("No un-analysed diagram chunks found. Nothing to do.")
        return

    logger.info(f"Found {len(pending)} diagram chunk(s) without Gemini analysis.")

    # ── 3. Initialise helpers (loads embedding model + Gemini client once) ─
    gemini_api_key = config.get('gemini.api_key') or ''
    if not gemini_api_key:
        logger.error(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY in env/.env and try again."
        )
        sys.exit(1)

    logger.info("Initialising PDF parser (Gemini) and embedding model ...")
    parser = PDFParser(config)
    vs = VectorStore(config)

    updated = 0
    skipped = 0

    # ── 4. Process each pending chunk ──────────────────────────────────────
    for chunk_idx, chunk in pending:
        diagram_id = chunk.metadata.get('diagram_id', '')
        caption = chunk.metadata.get('diagram_caption', '')
        if not caption:
            # Fallback: strip markdown bold markers from first line of content
            caption = chunk.content.split('\n')[0].strip('* ')

        location = _parse_diagram_id(diagram_id)
        if location is None:
            logger.warning(
                f"Cannot parse diagram_id '{diagram_id}' — "
                "only 'diagram_pN_iM' format is supported. Skipping."
            )
            skipped += 1
            continue

        page_num, img_num = location
        logger.info(f"Retrying diagram '{diagram_id}'  (page {page_num}, image {img_num}) ...")

        # Re-extract the image bytes from the original PDF
        image_bytes = _extract_image_bytes(pdf_path, page_num, img_num)
        if image_bytes is None:
            skipped += 1
            continue

        # Stop early if Gemini rate-limits again
        if parser._gemini_rate_limited:
            logger.warning("Gemini rate limit hit again. Stopping retries for now.")
            break

        # Call Gemini
        result = parser._analyze_image_with_gemini(image_bytes, caption)
        description = result.get('description', '')
        extracted_text = result.get('extracted_text', '')

        if not description and not extracted_text:
            logger.info(f"  Gemini returned no output for {diagram_id}. Skipping.")
            skipped += 1
            continue

        # ── 5. Rebuild chunk content (same format as _chunk_diagrams) ──────
        content_parts = [f"**{caption}**"]
        if description:
            content_parts.append(f"\n{description}")
        if extracted_text.strip() and extracted_text not in description:
            content_parts.append(f"\n**Key Elements**: {extracted_text}")
        content_parts.append(f"\n*Located on page {page_num}*")
        new_content = "\n".join(content_parts)

        chunk.content = new_content
        chunk.metadata['has_relationship_analysis'] = True
        chunk.metadata['has_extracted_data'] = bool(extracted_text.strip())

        # ── 6. Re-embed the updated chunk ───────────────────────────────────
        context_string = chunk.get_context_string()
        new_embedding = vs.get_embedding(context_string).astype(np.float32)
        embeddings[chunk_idx] = new_embedding

        updated += 1
        logger.info(f"  Updated chunk {chunk.chunk_id}.")

    # ── 7. Persist changes ─────────────────────────────────────────────────
    logger.info(f"\nRetry complete: {updated} updated, {skipped} skipped.")

    if updated == 0:
        logger.info("No chunks were updated — index left unchanged.")
        return

    logger.info("Rebuilding FAISS index with updated embeddings ...")
    dimension = embeddings.shape[1]
    norm_embeddings = embeddings.copy()
    faiss.normalize_L2(norm_embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(norm_embeddings)

    faiss.write_index(index, str(index_file))
    np.save(embeddings_file, embeddings)
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunks, f)

    logger.info(
        f"Saved updated FAISS index ({index.ntotal} vectors), "
        "embeddings.npy, and chunks.pkl."
    )
    logger.info("Done. All retried diagrams now have full Gemini analysis in the vector store.")


if __name__ == '__main__':
    main()
