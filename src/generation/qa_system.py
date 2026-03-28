"""
Main QA System interface for NASA Technical Manual.
Combines retrieval and generation for question answering with precise citations.
Supports both Groq (free tier) and OpenAI models.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

from src.config import get_config
from src.ingestion.vector_store import VectorStore
from src.retriever.retriever import MultiHopRetriever, RetrievalContext


logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a citation for an answer."""
    section_number: str
    section_title: str
    page_start: int
    page_end: int
    chunk_type: str
    confidence: float
    
    def format(self, style: str = 'section_page') -> str:
        """Format citation in specified style."""
        if style == 'section_page':
            if self.page_end != self.page_start:
                return f"Section {self.section_number}, pages {self.page_start}-{self.page_end}"
            else:
                return f"Section {self.section_number}, page {self.page_start}"
        elif style == 'paragraph':
            return f"§{self.section_number} (p. {self.page_start})"
        else:
            return f"[{self.section_number}]"


@dataclass
class CitationVerification:
    """Verification result for a single citation found in the LLM answer."""
    section_number: str          # e.g. "6.7"
    page_cited: Optional[int]    # page number the LLM mentioned, or None
    status: str                  # 'verified' | 'partial' | 'not_found'
    grounding_score: float       # 0–1 word-overlap between claim and chunk
    chunk_found: bool            # a chunk with this section exists
    page_in_range: bool          # cited page falls in chunk's page range
    matched_chunk_id: Optional[str]  # chunk_id of best matching chunk


@dataclass
class QAResult:
    """Represents a question-answering result."""
    query: str
    answer: str
    citations: List[Citation]
    confidence: float
    context_used: int
    metadata: Dict[str, Any]
    citation_verification: List['CitationVerification'] = None

    def __post_init__(self):
        if self.citation_verification is None:
            self.citation_verification = []

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'answer': self.answer,
            'citations': [asdict(c) for c in self.citations],
            'confidence': self.confidence,
            'context_used': self.context_used,
            'metadata': self.metadata,
            'citation_verification': [asdict(v) for v in self.citation_verification],
        }


class TechnicalQASystem:
    """Technical documentation QA system."""
    
    def __init__(self, vector_store_path: Optional[str] = None, config=None):
        """
        Initialize QA system.
        
        Args:
            vector_store_path: Path to load vector store from
            config: Configuration instance
        """
        self.config = config or get_config()
        self.generation_config = self.config.get_generation_config()
        
        # Determine LLM provider
        llm_provider = self.config.get('llm_provider', 'groq')
        self.llm_provider = llm_provider
        
        # Initialize LLM client
        if llm_provider == 'groq':
            if not HAS_GROQ:
                raise ImportError("groq not installed. Install with: pip install groq")
            groq_config = self.config.get('groq', {})
            api_key = groq_config.get('api_key')
            if not api_key:
                raise ValueError("Groq API key required. Set GROQ_API_KEY in config or .env")
            self.client = Groq(api_key=api_key)
            self.llm_model = groq_config.get('llm_model', 'openai/gpt-oss-120b')
            self.temperature = groq_config.get('temperature', 0.1)
            self.max_tokens = groq_config.get('max_tokens', 2000)
            logger.info(f"Using Groq model: {self.llm_model}")
        elif llm_provider == 'openai':
            if not HAS_OPENAI:
                raise ImportError("openai not installed. Install with: pip install openai")
            openai_config = self.config.get_openai_config()
            api_key = openai_config.get('api_key')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in config or .env")
            self.client = OpenAI(api_key=api_key)
            self.llm_model = openai_config.get('llm_model', 'gpt-4-turbo-preview')
            self.temperature = openai_config.get('temperature', 0.1)
            self.max_tokens = openai_config.get('max_tokens', 2000)
            logger.info(f"Using OpenAI model: {self.llm_model}")
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
        
        # Load vector store
        logger.info("Loading vector store...")
        self.vector_store = VectorStore(self.config)
        store_path = vector_store_path or self.config.get('paths.vector_store')
        self.vector_store.load(store_path)
        logger.info(f"Loaded vector store with {len(self.vector_store.chunks)} chunks")
        
        # Initialize retriever
        self.retriever = MultiHopRetriever(self.vector_store, self.config)
        
        # System prompt
        self.system_prompt = self.generation_config.get('system_prompt', '')
        
        logger.info("QA System initialized and ready")
    
    def ask(self, query: str, include_context: bool = False) -> Dict[str, Any]:
        """
        Answer a question about the technical manual.
        
        Args:
            query: User question
            include_context: Whether to include retrieved context in result
            
        Returns:
            QA result dictionary
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant context
        retrieval_context = self.retriever.retrieve(query)
        
        # Step 2: Format context for LLM
        formatted_context = self.retriever.format_context_for_llm(retrieval_context)
        
        # Step 3: Generate answer using LLM
        answer, confidence = self._generate_answer(query, formatted_context)
        
        # Step 4: Extract citations
        citations = self._extract_citations(retrieval_context, confidence)

        # Step 5: Verify citations against the chunk store
        verification = self._verify_citations(answer, self.vector_store.chunks)

        # Step 6: Build result
        result = QAResult(
            query=query,
            answer=answer,
            citations=citations,
            confidence=confidence,
            context_used=len(retrieval_context.all_results),
            metadata={
                'multi_hop_used': len(retrieval_context.cross_reference_results) > 0,
                'parent_sections_used': len(retrieval_context.parent_section_results) > 0,
                'model': self.llm_model
            },
            citation_verification=verification,
        )
        
        result_dict = result.to_dict()
        if include_context:
            result_dict['retrieved_context'] = [
                {
                    'section': r.section_number,
                    'title': r.section_title,
                    'content': r.content,
                    'score': r.score
                }
                for r in retrieval_context.all_results
            ]
        
        return result_dict
    
    def _generate_answer(self, query: str, context: str) -> tuple[str, float]:
        """
        Generate answer using LLM.
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Build prompt
        prompt = f"""Based on the following context from the NASA Systems Engineering Handbook, please answer the question.

Provide a clear, accurate answer with specific citations in the format: (Section X.Y.Z, page N).

If the information spans multiple sections, synthesize the answer and cite all relevant sources.

If you cannot find the answer in the provided context, say "I cannot find this information in the provided sections of the handbook."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Estimate confidence based on response quality
            confidence = self._estimate_confidence(answer, context)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}", 0.0
    
    def _estimate_confidence(self, answer: str, context: str) -> float:
        """
        Estimate confidence score for the answer.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Confidence score (0-1)
        """
        # Start with base confidence
        confidence = 0.5
        
        # Boost if answer contains citations
        if 'Section' in answer and 'page' in answer:
            confidence += 0.2
        
        # Boost if answer is substantial
        if len(answer) > 100:
            confidence += 0.1
        
        # Reduce if answer indicates uncertainty
        uncertainty_phrases = [
            "cannot find",
            "don't know",
            "unclear",
            "not specified",
            "not mentioned"
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3
        
        # Ensure in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    # ------------------------------------------------------------------
    # Citation verification
    # ------------------------------------------------------------------

    def _verify_citations(
        self,
        answer: str,
        chunks: list,
    ) -> List[CitationVerification]:
        """
        Parse citations from the LLM answer and verify each one against
        the chunk store.

        Strategy
        --------
        1. Extract every ``(Section X.Y.Z, page N)`` reference in the answer.
        2. For each reference:
           a. Find all chunks whose section_number starts with the cited section.
           b. Extract the sentence that contains the citation — this is the
              *claim* the LLM is attributing to that source.
           c. Compute word-overlap (Jaccard on 4+ char words) between the
              claim and each candidate chunk's content.  Pick the best.
           d. Assign status:
              - ``verified``  : chunk found + page in range + overlap ≥ 0.15
              - ``partial``   : chunk found but page mismatch or overlap < 0.15
              - ``not_found`` : no chunk with that section number exists
        3. Deduplicate by section_number (keep highest grounding_score).
        """
        import re

        # Pattern: "Section 6.7" / "Section 6.7.1" with optional ", page N"
        cite_re = re.compile(
            r'[Ss]ection\s+(\d{1,2}(?:\.\d{1,2}){0,3})'
            r'(?:[,\s]+(?:p(?:age|\.)?\.?\s*)(\d+))?',
        )

        # Split answer into sentences for claim extraction
        sentence_re = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_re.split(answer)

        def _claim_for_citation(sec: str, page: Optional[int]) -> str:
            """Return the sentence(s) that contain this citation."""
            label = f'Section {sec}'
            for sent in sentences:
                if label in sent:
                    return sent
            return answer  # fallback: whole answer

        def _word_overlap(text_a: str, text_b: str) -> float:
            """Jaccard similarity on words with 4+ chars (case-insensitive)."""
            words_a = set(w.lower() for w in re.findall(r'\b\w{4,}\b', text_a))
            words_b = set(w.lower() for w in re.findall(r'\b\w{4,}\b', text_b))
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / len(words_a | words_b)

        # Build a section → [chunks] lookup once
        section_index: Dict[str, list] = {}
        for chunk in chunks:
            sec = chunk.section_number
            section_index.setdefault(sec, []).append(chunk)

        seen: Dict[str, CitationVerification] = {}  # section_number → best result

        for m in cite_re.finditer(answer):
            sec_num = m.group(1)
            page_cited = int(m.group(2)) if m.group(2) else None

            # Candidate chunks: exact section OR any sub-section prefix match
            candidates = []
            for sec, clist in section_index.items():
                if sec == sec_num or sec.startswith(sec_num + '.'):
                    candidates.extend(clist)

            if not candidates:
                result = CitationVerification(
                    section_number=sec_num,
                    page_cited=page_cited,
                    status='not_found',
                    grounding_score=0.0,
                    chunk_found=False,
                    page_in_range=False,
                    matched_chunk_id=None,
                )
            else:
                claim = _claim_for_citation(sec_num, page_cited)
                best_score = -1.0
                best_chunk = None
                best_page_ok = False

                for chunk in candidates:
                    score = _word_overlap(claim, chunk.content)
                    page_ok = (
                        page_cited is not None
                        and chunk.page_start <= page_cited <= (chunk.page_end or chunk.page_start)
                    )
                    # Prefer chunks where page also matches
                    effective = score + (0.1 if page_ok else 0.0)
                    if effective > best_score:
                        best_score = effective
                        best_chunk = chunk
                        best_page_ok = page_ok

                raw_score = _word_overlap(claim, best_chunk.content)
                if raw_score >= 0.15 and (page_cited is None or best_page_ok):
                    status = 'verified'
                elif best_chunk is not None:
                    status = 'partial'
                else:
                    status = 'not_found'

                result = CitationVerification(
                    section_number=sec_num,
                    page_cited=page_cited,
                    status=status,
                    grounding_score=round(raw_score, 4),
                    chunk_found=True,
                    page_in_range=best_page_ok,
                    matched_chunk_id=best_chunk.chunk_id,
                )

            # Keep the best result per section
            existing = seen.get(sec_num)
            if existing is None or result.grounding_score > existing.grounding_score:
                seen[sec_num] = result

        return list(seen.values())

    def _extract_citations(
        self,
        retrieval_context: RetrievalContext,
        confidence: float
    ) -> List[Citation]:
        """
        Extract citations from retrieval context.
        
        Args:
            retrieval_context: Retrieved context
            confidence: Answer confidence
            
        Returns:
            List of citations
        """
        citations = []
        seen_sections = set()
        
        # Include top results that meet confidence threshold
        min_confidence = self.generation_config.get('min_confidence_threshold', 0.3)
        
        for result in retrieval_context.all_results:
            section_num = result.section_number
            
            # Skip if already included or below threshold
            if section_num in seen_sections or result.score < min_confidence:
                continue
            
            seen_sections.add(section_num)
            
            citation = Citation(
                section_number=result.section_number,
                section_title=result.section_title,
                page_start=result.page_start,
                page_end=result.page_end,
                chunk_type=result.chunk_type,
                confidence=result.score
            )
            citations.append(citation)
        
        # Sort by confidence
        citations.sort(key=lambda x: x.confidence, reverse=True)
        
        return citations
    
    def format_answer(self, result: Dict[str, Any]) -> str:
        """
        Format QA result for display.
        
        Args:
            result: QA result dictionary
            
        Returns:
            Formatted string
        """
        output = []
        
        # Question
        output.append(f"Q: {result['query']}\n")
        
        # Answer
        output.append(f"A: {result['answer']}\n")
        
        # Citations
        if result['citations']:
            output.append("Sources:")
            citation_format = self.generation_config.get('citation_format', 'section_page')
            
            for i, citation_dict in enumerate(result['citations'], 1):
                citation = Citation(**citation_dict)
                formatted_citation = citation.format(citation_format)
                output.append(f"  {i}. {formatted_citation}")
                output.append(f"     ({citation.section_title})")
                if self.generation_config.get('include_confidence', True):
                    output.append(f"     Confidence: {citation.confidence:.2f}")
        
        # Metadata
        if result.get('metadata'):
            output.append(f"\nContext used: {result['context_used']} chunks")
            if result['metadata'].get('multi_hop_used'):
                output.append("✓ Multi-hop cross-references resolved")
        
        return '\n'.join(output)
    
    def batch_ask(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.
        
        Args:
            queries: List of questions
            
        Returns:
            List of QA results
        """
        results = []
        for query in queries:
            result = self.ask(query)
            results.append(result)
        return results


def main():
    """Main function for interactive QA."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NASA Handbook QA System')
    parser.add_argument('--query', type=str, help='Single query to answer')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--test', action='store_true', help='Run test queries')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize QA system
    print("Initializing QA system...")
    qa_system = TechnicalQASystem()
    print("✓ QA system ready!\n")
    
    if args.query:
        # Single query
        result = qa_system.ask(args.query)
        print(qa_system.format_answer(result))
    
    elif args.test:
        # Test queries
        test_queries = [
            "What are the entry criteria for PDR?",
            "How does risk management feed into the technical review process?",
            "What is TRL and what are its levels?",
            "What does the systems engineering process flow look like?",
            "What should a verification plan include?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'=' * 80}")
            print(f"Test Query {i}/{len(test_queries)}")
            print(f"{'=' * 80}\n")
            
            result = qa_system.ask(query)
            print(qa_system.format_answer(result))
            print()
    
    elif args.interactive:
        # Interactive mode
        print("Interactive QA mode (type 'quit' to exit)\n")
        
        while True:
            try:
                query = input("\nQ: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                result = qa_system.ask(query)
                print(f"\n{qa_system.format_answer(result)}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")
    
    else:
        print("Usage:")
        print("  --query 'your question'  : Ask a single question")
        print("  --interactive           : Interactive Q&A mode")
        print("  --test                  : Run test queries")


if __name__ == '__main__':
    main()
