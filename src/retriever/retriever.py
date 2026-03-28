"""
Retriever module with multi-hop cross-reference resolution.
Implements hybrid retrieval combining vector search and keyword matching.
"""
import logging
import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

from src.config import get_config
from src.ingestion.vector_store import VectorStore, SearchResult
from src.ingestion.chunker import DocumentChunk


logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Represents retrieved context for answering a query."""
    query: str
    primary_results: List[SearchResult]
    cross_reference_results: List[SearchResult]
    parent_section_results: List[SearchResult]
    all_results: List[SearchResult]
    metadata: Dict


class MultiHopRetriever:
    """Retriever with multi-hop cross-reference resolution."""
    
    def __init__(self, vector_store: VectorStore, config=None):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            config: Configuration instance
        """
        self.vector_store = vector_store
        self.config = config or get_config()
        self.retrieval_config = self.config.get_retrieval_config()
        
        self.top_k = self.retrieval_config.get('top_k', 5)
        self.rerank = self.retrieval_config.get('rerank', True)
        self.rerank_top_k = self.retrieval_config.get('rerank_top_k', 10)
        self.enable_multi_hop = self.retrieval_config.get('enable_multi_hop', True)
        self.max_hops = self.retrieval_config.get('max_hops', 2)
        self.fetch_parent = self.retrieval_config.get('fetch_parent_sections', True)
        
        # Hybrid search weights
        self.vector_weight = self.retrieval_config.get('vector_weight', 0.7)
        self.keyword_weight = self.retrieval_config.get('keyword_weight', 0.3)
        
        logger.info(f"Initialized retriever: top_k={self.top_k}, multi_hop={self.enable_multi_hop}")
    
    def retrieve(self, query: str) -> RetrievalContext:
        """
        Retrieve relevant context for a query with multi-hop resolution.
        
        Args:
            query: User query
            
        Returns:
            RetrievalContext with all retrieved results
        """
        logger.info(f"Retrieving context for query: {query}")
        
        # Step 1: Primary vector search
        primary_results = self._vector_search(query)
        logger.info(f"Primary vector search: {len(primary_results)} results")
        
        # Step 2: Keyword boosting for section numbers and acronyms
        if self.retrieval_config.get('strategy') == 'hybrid':
            primary_results = self._apply_keyword_boost(query, primary_results)
        
        # Step 3: Multi-hop cross-reference resolution
        cross_ref_results = []
        if self.enable_multi_hop:
            cross_ref_results = self._resolve_cross_references(primary_results)
            logger.info(f"Cross-reference resolution: {len(cross_ref_results)} additional results")
        
        # Step 4: Fetch parent sections for context
        parent_results = []
        if self.fetch_parent:
            parent_results = self._fetch_parent_sections(primary_results)
            logger.info(f"Parent section fetch: {len(parent_results)} results")
        
        # Step 5: Combine and deduplicate results
        all_results = self._combine_and_deduplicate(
            primary_results,
            cross_ref_results,
            parent_results
        )
        
        # Step 6: Rerank if enabled
        if self.rerank:
            all_results = self._rerank_results(query, all_results)
        
        # Take top K final results
        final_results = all_results[:self.top_k]
        
        context = RetrievalContext(
            query=query,
            primary_results=primary_results,
            cross_reference_results=cross_ref_results,
            parent_section_results=parent_results,
            all_results=final_results,
            metadata={
                'total_retrieved': len(all_results),
                'multi_hop_enabled': self.enable_multi_hop,
                'parent_fetch_enabled': self.fetch_parent
            }
        )
        
        logger.info(f"Final retrieval: {len(final_results)} results")
        return context
    
    def _vector_search(self, query: str) -> List[SearchResult]:
        """Perform vector similarity search."""
        # Search with more results initially for later filtering
        search_k = self.rerank_top_k if self.rerank else self.top_k
        return self.vector_store.search(query, top_k=search_k)
    
    def _apply_keyword_boost(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Boost results that match keywords (section numbers, acronyms).
        
        Args:
            query: User query
            results: Search results
            
        Returns:
            Results with boosted scores
        """
        # Extract section numbers from query (e.g., "6.3", "Section 6")
        section_pattern = r'(?:Section\s+)?(\d+(?:\.\d+)*)'
        section_matches = re.findall(section_pattern, query, re.IGNORECASE)
        
        # Extract potential acronyms (2+ capital letters)
        acronym_pattern = r'\b([A-Z]{2,})\b'
        acronym_matches = re.findall(acronym_pattern, query)
        
        # Boost scores if matches found
        boosted_results = []
        for result in results:
            boost_factor = 1.0
            
            # Boost if section number matches
            for section in section_matches:
                if result.section_number.startswith(section):
                    boost_factor += 0.3
            
            # Boost if acronym found in content
            for acronym in acronym_matches:
                if acronym in result.content:
                    boost_factor += 0.2
            
            # Apply boost
            boosted_score = result.score * (self.vector_weight + self.keyword_weight * boost_factor)
            
            boosted_result = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=boosted_score,
                section_number=result.section_number,
                section_title=result.section_title,
                page_start=result.page_start,
                page_end=result.page_end,
                metadata=result.metadata,
                chunk_type=result.chunk_type
            )
            boosted_results.append(boosted_result)
        
        # Re-sort by boosted scores
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        
        return boosted_results
    
    def _resolve_cross_references(
        self,
        results: List[SearchResult],
        visited: Optional[Set[str]] = None,
        depth: int = 0
    ) -> List[SearchResult]:
        """
        Recursively resolve cross-references.
        
        Args:
            results: Current search results
            visited: Set of already visited sections
            depth: Current recursion depth
            
        Returns:
            Additional results from cross-references
        """
        if depth >= self.max_hops:
            return []
        
        if visited is None:
            visited = set()
        
        # Mark current sections as visited
        for result in results:
            visited.add(result.section_number)
        
        cross_ref_results = []
        
        # Find outgoing cross-references
        for result in results:
            cross_refs = result.metadata.get('cross_references', {})
            outgoing = cross_refs.get('outgoing', [])
            
            for target_section in outgoing:
                if target_section in visited:
                    continue
                
                # Fetch chunks from target section
                target_chunks = self.vector_store.get_chunks_by_section(target_section)
                
                for chunk in target_chunks:
                    # Create search result with lower score (discounted by hop distance)
                    discount = 0.7 ** (depth + 1)
                    
                    cross_ref_result = SearchResult(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        score=0.5 * discount,  # Base score for cross-ref
                        section_number=chunk.section_number,
                        section_title=chunk.section_title,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        metadata=chunk.metadata,
                        chunk_type=chunk.chunk_type
                    )
                    cross_ref_results.append(cross_ref_result)
                    visited.add(target_section)
        
        # Recursive multi-hop (if we found new references)
        if cross_ref_results and depth + 1 < self.max_hops:
            deeper_results = self._resolve_cross_references(
                cross_ref_results,
                visited,
                depth + 1
            )
            cross_ref_results.extend(deeper_results)
        
        return cross_ref_results
    
    def _fetch_parent_sections(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Fetch parent sections for context.
        
        Args:
            results: Current search results
            
        Returns:
            Parent section results
        """
        parent_results = []
        seen_sections = set()
        
        for result in results:
            # Skip if we've already processed this section
            section_num = result.section_number
            if section_num in seen_sections:
                continue
            seen_sections.add(section_num)
            
            # Get parent section number
            parts = section_num.split('.')
            if len(parts) <= 1:
                continue  # Top-level section, no parent
            
            parent_num = '.'.join(parts[:-1])
            if parent_num in seen_sections:
                continue
            seen_sections.add(parent_num)
            
            # Fetch parent chunks
            parent_chunks = self.vector_store.get_chunks_by_section(parent_num)
            
            for chunk in parent_chunks:
                parent_result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=result.score * 0.6,  # Discounted score for parent
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    metadata=chunk.metadata,
                    chunk_type=chunk.chunk_type
                )
                parent_results.append(parent_result)
        
        return parent_results
    
    def _combine_and_deduplicate(
        self,
        *result_lists: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Combine multiple result lists and deduplicate.
        
        Args:
            result_lists: Variable number of result lists
            
        Returns:
            Combined and deduplicated results
        """
        combined = {}
        
        for results in result_lists:
            for result in results:
                chunk_id = result.chunk_id
                
                # Keep highest score if duplicate
                if chunk_id in combined:
                    if result.score > combined[chunk_id].score:
                        combined[chunk_id] = result
                else:
                    combined[chunk_id] = result
        
        # Convert back to list and sort by score
        all_results = list(combined.values())
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank results using vector similarity + keyword signals.
        
        Computes fresh cosine similarity between the query and each chunk,
        then combines it with keyword-based boosts. This is especially useful
        for parent and cross-reference chunks whose initial scores are
        discounted and may not reflect true relevance to the query.
        
        Args:
            query: User query
            results: Search results to rerank
            
        Returns:
            Reranked results
        """
        import numpy as np
        import faiss
        
        # Compute query embedding once
        query_embedding = self.vector_store.get_embedding(query).reshape(1, -1)
        if self.vector_store.distance_metric == 'cosine':
            faiss.normalize_L2(query_embedding)
        
        reranked = []
        
        for result in results:
            # --- Vector similarity score ---
            # Compute fresh cosine similarity between query and chunk content
            chunk_embedding = self.vector_store.get_embedding(result.content).reshape(1, -1)
            if self.vector_store.distance_metric == 'cosine':
                faiss.normalize_L2(chunk_embedding)
            
            vector_sim = float(np.dot(query_embedding, chunk_embedding.T)[0][0])
            
            # --- Keyword boost factor ---
            keyword_boost = 1.0
            
            # Boost for exact phrase matches
            if query.lower() in result.content.lower():
                keyword_boost *= 1.2
            
            # Boost for title matches
            if query.lower() in result.section_title.lower():
                keyword_boost *= 1.15

            # Boost tables whose caption has strong word-overlap with the query
            if result.chunk_type == 'table':
                cap_match = re.match(r'\*\*([^*]+)\*\*', result.content.strip())
                if cap_match:
                    cap_words = set(re.findall(r'\b\w{4,}\b', cap_match.group(1).lower()))
                    qry_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
                    if cap_words and qry_words:
                        overlap = len(cap_words & qry_words) / len(qry_words)
                        if overlap >= 0.35:
                            keyword_boost *= (1.0 + overlap * 0.6)

            # Boost tables for "what are" questions
            if result.chunk_type == 'table' and 'what are' in query.lower():
                keyword_boost *= 1.1
            
            # Boost diagrams for "show" or "look like" questions
            if result.chunk_type == 'diagram_ref' and ('show' in query.lower() or 'look like' in query.lower()):
                keyword_boost *= 1.15
            
            # --- Combined score ---
            # Blend vector similarity (70%) with keyword-boosted original score (30%)
            combined_score = (0.7 * vector_sim + 0.3 * result.score) * keyword_boost
            
            reranked_result = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=combined_score,
                section_number=result.section_number,
                section_title=result.section_title,
                page_start=result.page_start,
                page_end=result.page_end,
                metadata=result.metadata,
                chunk_type=result.chunk_type
            )
            reranked.append(reranked_result)
        
        # Sort by new scores
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked
    
    def format_context_for_llm(self, context: RetrievalContext) -> str:
        """
        Format retrieved context for LLM consumption.
        
        Args:
            context: Retrieval context
            
        Returns:
            Formatted context string
        """
        formatted_parts = []
        
        for i, result in enumerate(context.all_results, 1):
            part = f"[Context {i}]\n"
            part += f"Section: {result.section_number} - {result.section_title}\n"
            part += f"Page: {result.page_start}"
            if result.page_end != result.page_start:
                part += f"-{result.page_end}"
            part += f"\nType: {result.chunk_type}\n"
            part += f"Relevance Score: {result.score:.3f}\n\n"
            part += result.content
            part += "\n" + "=" * 80 + "\n"
            
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)


def main():
    """Main function for testing retriever."""
    import json
    from src.vector_store import VectorStore
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = get_config()
    
    # Load vector store
    vector_store = VectorStore(config)
    vector_store_path = config.get('paths.vector_store')
    vector_store.load(vector_store_path)
    
    print(f"✓ Loaded vector store with {len(vector_store.chunks)} chunks\n")
    
    # Initialize retriever
    retriever = MultiHopRetriever(vector_store, config)
    
    # Test queries
    test_queries = [
        "What are the entry criteria for PDR?",
        "How does risk management feed into technical review?",
        "What is TRL and what are its levels?",
        "What does the systems engineering process flow look like?",
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}\n")
        
        context = retriever.retrieve(query)
        
        print(f"Retrieved {len(context.all_results)} total results:")
        print(f"  - Primary: {len(context.primary_results)}")
        print(f"  - Cross-references: {len(context.cross_reference_results)}")
        print(f"  - Parent sections: {len(context.parent_section_results)}")
        print()
        
        for i, result in enumerate(context.all_results[:3], 1):
            print(f"{i}. Section {result.section_number}: {result.section_title}")
            print(f"   Score: {result.score:.3f} | Type: {result.chunk_type} | Page: {result.page_start}")
            print(f"   {result.content[:150]}...")
            print()


if __name__ == '__main__':
    main()
