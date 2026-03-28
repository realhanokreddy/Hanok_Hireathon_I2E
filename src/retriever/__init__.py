"""
Retrieval module for NASA QA System.
Handles multi-hop retrieval with cross-reference resolution.
"""

from src.retriever.retriever import MultiHopRetriever, RetrievalContext

__all__ = [
    'MultiHopRetriever',
    'RetrievalContext',
]
