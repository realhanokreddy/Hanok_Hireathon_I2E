"""
Ingestion module for NASA QA System.
Handles PDF parsing, metadata extraction, chunking, and vector store creation.
"""

from src.ingestion.pdf_parser import PDFParser, ParsedDocument, ParsedSection, ParsedTable, ParsedDiagram
from src.ingestion.metadata_extractor import MetadataExtractor, AcronymDefinition, CrossReference, SectionHierarchy
from src.ingestion.chunker import HierarchicalChunker, DocumentChunk
from src.ingestion.vector_store import VectorStore, SearchResult

__all__ = [
    'PDFParser',
    'ParsedDocument',
    'ParsedSection',
    'ParsedTable',
    'ParsedDiagram',
    'MetadataExtractor',
    'AcronymDefinition',
    'CrossReference',
    'SectionHierarchy',
    'HierarchicalChunker',
    'DocumentChunk',
    'VectorStore',
    'SearchResult',
]
