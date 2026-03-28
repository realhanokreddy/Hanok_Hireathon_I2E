"""NASA Technical Manual QA System - Package initialization."""

__version__ = "1.0.0"
__author__ = "NASA QA System Team"

from src.config import get_config
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.vector_store import VectorStore
from src.retriever.retriever import MultiHopRetriever
from src.generation.qa_system import TechnicalQASystem

__all__ = [
    'get_config',
    'PDFParser',
    'MetadataExtractor',
    'HierarchicalChunker',
    'VectorStore',
    'MultiHopRetriever',
    'TechnicalQASystem',
]
