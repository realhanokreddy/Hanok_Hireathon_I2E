"""
Hierarchical chunking module for technical documents.
Respects section boundaries and maintains document structure.
"""
import logging
import tiktoken
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import json

from src.config import get_config
from src.ingestion.pdf_parser import ParsedDocument, ParsedSection, ParsedTable
from src.ingestion.metadata_extractor import MetadataExtractor, SectionHierarchy, AcronymDefinition


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata."""
    chunk_id: str
    content: str
    chunk_type: str  # 'text', 'table', 'diagram_ref', 'code'
    section_number: str
    section_title: str
    section_level: int
    parent_section: Optional[str]
    section_path: List[str]
    page_start: int
    page_end: int
    token_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_context_string(self) -> str:
        """Get a context string for embedding."""
        import re

        # For tables, extract the bold caption line and use it as primary context.
        # Format in content: **TABLE X-Y Some Title** (Rows N-M)
        table_caption = None
        if self.chunk_type == 'table':
            m = re.match(r'\*\*([^*]+)\*\*', self.content.strip())
            if m:
                table_caption = m.group(1).strip()

        context_parts = []
        if self.section_path:
            context_parts.append(f"Chapter {self.section_path[0]}")
        if table_caption:
            context_parts.append(f"Table: {table_caption}")
        context_parts.append(f"Section {self.section_number}: {self.section_title}")
        context_parts.append(f"Type: {self.chunk_type}")

        # Add parent section context
        if self.parent_section and 'parent_title' in self.metadata:
            context_parts.append(f"Under: {self.metadata['parent_title']}")

        context = " | ".join([p for p in context_parts if p])
        return f"{context}\n\n{self.content}"


class HierarchicalChunker:
    """Chunk documents while preserving hierarchical structure."""
    
    def __init__(self, config=None):
        """
        Initialize hierarchical chunker.
        
        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self.chunking_config = self.config.get_chunking_config()
        
        self.chunk_size = self.chunking_config.get('chunk_size', 800)
        self.chunk_overlap = self.chunking_config.get('chunk_overlap', 150)
        self.min_chunk_size = self.chunking_config.get('min_chunk_size', 200)
        self.respect_boundaries = self.chunking_config.get('respect_section_boundaries', True)
        self.include_parent = self.chunking_config.get('include_parent_context', True)
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoder: {e}. Using approximate token count.")
            self.tokenizer = None
        
        logger.info(f"Initialized chunker: size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4
    
    def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        metadata: Dict,
        acronyms: Dict[str, AcronymDefinition] = None
    ) -> List[DocumentChunk]:
        """
        Chunk document hierarchically.
        
        Args:
            parsed_doc: Parsed document instance
            metadata: Extracted metadata
            acronyms: Dictionary of acronyms
            
        Returns:
            List of document chunks
        """
        logger.info("Chunking document hierarchically")
        
        chunks = []
        chunk_counter = 0
        
        section_hierarchy = metadata.get('section_hierarchy', {})
        cross_refs = metadata.get('cross_references', [])
        
        # Chunk each section
        for section in parsed_doc.sections:
            section_chunks = self._chunk_section(
                section,
                section_hierarchy,
                cross_refs,
                acronyms,
                chunk_counter
            )
            chunks.extend(section_chunks)
            chunk_counter += len(section_chunks)
        
        # Chunk tables separately
        table_chunks = self._chunk_tables(
            parsed_doc.tables,
            section_hierarchy,
            acronyms,
            chunk_counter
        )
        chunks.extend(table_chunks)
        chunk_counter += len(table_chunks)
        
        # Add diagram references
        diagram_chunks = self._chunk_diagrams(
            parsed_doc.diagrams,
            section_hierarchy,
            acronyms,
            chunk_counter
        )
        chunks.extend(diagram_chunks)

        # Re-order chunks by document position: page_start asc, then type order
        # (text < table < diagram_ref) so text that precedes a table stays before it.
        type_order = {'text': 0, 'table': 1, 'diagram_ref': 2, 'code': 0}
        chunks.sort(key=lambda c: (c.page_start, type_order.get(c.chunk_type, 9)))

        # Reassign sequential chunk_ids to reflect the new order
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"chunk_{i}"

        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _chunk_section(
        self,
        section: ParsedSection,
        hierarchy: Dict[str, SectionHierarchy],
        cross_refs: List,
        acronyms: Optional[Dict[str, AcronymDefinition]],
        start_id: int
    ) -> List[DocumentChunk]:
        """Chunk a single section."""
        chunks = []
        
        # Get section hierarchy info
        section_info = hierarchy.get(section.section_number, {})
        if isinstance(section_info, SectionHierarchy):
            section_path = section_info.path
            parent_section = section_info.parent
        else:
            section_path = [section.section_number]
            parent_section = section.parent_section
        
        # Get parent title for context
        parent_title = ""
        if parent_section:
            for sec_num, sec_info in hierarchy.items():
                if sec_num == parent_section:
                    parent_title = sec_info.title if isinstance(sec_info, SectionHierarchy) else ""
                    break
        
        # Prepare content
        content = section.content
        if not content or len(content.strip()) == 0:
            return chunks
        
        # Optionally expand acronyms
        if acronyms and self.config.get_metadata_config().get('extract_acronyms'):
            # Don't expand in original, but include in metadata
            pass
        
        # Add section title as context
        full_content = f"# {section.title}\n\n{content}"
        
        # Count tokens
        total_tokens = self.count_tokens(full_content)
        
        if total_tokens <= self.chunk_size:
            # Single chunk for the section
            chunk_metadata = {
                'parent_title': parent_title,
                'has_subsections': len(section.subsections) > 0,
                'cross_references': self._get_section_cross_refs(section.section_number, cross_refs)
            }
            
            # Add acronyms found in this chunk
            if acronyms:
                found_acronyms = self._extract_acronyms_from_text(full_content, acronyms)
                if found_acronyms:
                    chunk_metadata['acronyms'] = found_acronyms
            
            chunk = DocumentChunk(
                chunk_id=f"chunk_{start_id}",
                content=full_content,
                chunk_type='text',
                section_number=section.section_number,
                section_title=section.title,
                section_level=section.level,
                parent_section=parent_section,
                section_path=section_path,
                page_start=section.page_start,
                page_end=section.page_end,
                token_count=total_tokens,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks with overlap
            section_chunks = self._split_text_with_overlap(
                full_content,
                section,
                section_path,
                parent_section,
                parent_title,
                cross_refs,
                acronyms,
                start_id
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_text_with_overlap(
        self,
        text: str,
        section: ParsedSection,
        section_path: List[str],
        parent_section: Optional[str],
        parent_title: str,
        cross_refs: List,
        acronyms: Optional[Dict[str, AcronymDefinition]],
        start_id: int
    ) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        chunks = []
        
        # Split by paragraphs first (preserve paragraph boundaries)
        paragraphs = text.split('\n\n')
        
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            if current_chunk_tokens + para_tokens > self.chunk_size and current_chunk_text:
                # Save current chunk
                chunk_metadata = {
                    'parent_title': parent_title,
                    'chunk_index': chunk_idx,
                    'is_continuation': chunk_idx > 0,
                    'cross_references': self._get_section_cross_refs(section.section_number, cross_refs)
                }
                
                # Add acronyms found in this chunk
                if acronyms:
                    found_acronyms = self._extract_acronyms_from_text(current_chunk_text, acronyms)
                    if found_acronyms:
                        chunk_metadata['acronyms'] = found_acronyms
                
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{start_id + chunk_idx}",
                    content=current_chunk_text.strip(),
                    chunk_type='text',
                    section_number=section.section_number,
                    section_title=section.title,
                    section_level=section.level,
                    parent_section=parent_section,
                    section_path=section_path,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    token_count=current_chunk_tokens,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                # Take last paragraph(s) as overlap
                overlap_text = self._get_overlap_text(current_chunk_text, self.chunk_overlap)
                current_chunk_text = overlap_text + "\n\n" + para
                current_chunk_tokens = self.count_tokens(current_chunk_text)
                chunk_idx += 1
            else:
                # Add to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
                current_chunk_tokens = self.count_tokens(current_chunk_text)
        
        # Add final chunk
        if current_chunk_text and current_chunk_tokens >= self.min_chunk_size:
            chunk_metadata = {
                'parent_title': parent_title,
                'chunk_index': chunk_idx,
                'is_continuation': chunk_idx > 0,
                'cross_references': self._get_section_cross_refs(section.section_number, cross_refs)
            }
            
            # Add acronyms found in this chunk
            if acronyms:
                found_acronyms = self._extract_acronyms_from_text(current_chunk_text, acronyms)
                if found_acronyms:
                    chunk_metadata['acronyms'] = found_acronyms
            
            chunk = DocumentChunk(
                chunk_id=f"chunk_{start_id + chunk_idx}",
                content=current_chunk_text.strip(),
                chunk_type='text',
                section_number=section.section_number,
                section_title=section.title,
                section_level=section.level,
                parent_section=parent_section,
                section_path=section_path,
                page_start=section.page_start,
                page_end=section.page_end,
                token_count=current_chunk_tokens,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get last N tokens from text for overlap."""
        if not text:
            return ""
        
        # Split by sentences (simple split)
        sentences = text.split('. ')
        
        # Take last few sentences that fit in overlap
        overlap_text = ""
        for sentence in reversed(sentences):
            sentence_with_period = sentence + '. ' if not sentence.endswith('.') else sentence
            test_text = sentence_with_period + overlap_text
            if self.count_tokens(test_text) <= overlap_tokens:
                overlap_text = test_text
            else:
                break
        
        return overlap_text.strip()
    
    def _extract_acronyms_from_text(
        self, 
        text: str, 
        acronyms: Optional[Dict[str, AcronymDefinition]]
    ) -> Dict[str, str]:
        """
        Extract acronyms present in text and return their definitions.
        
        Args:
            text: Text content to scan
            acronyms: Dictionary of all known acronyms
            
        Returns:
            Dictionary mapping acronyms found in text to their definitions
        """
        if not acronyms:
            return {}
        
        import re
        found_acronyms = {}
        
        # Pattern to find acronyms (2+ uppercase letters)
        acronym_pattern = r'\b([A-Z]{2,})\b'
        
        for match in re.finditer(acronym_pattern, text):
            acronym = match.group(1)
            if acronym in acronyms:
                acronym_obj = acronyms[acronym]
                found_acronyms[acronym] = acronym_obj.definition
        
        return found_acronyms
    
    def _chunk_tables(
        self,
        tables: List[ParsedTable],
        hierarchy: Dict,
        acronyms: Optional[Dict[str, AcronymDefinition]],
        start_id: int
    ) -> List[DocumentChunk]:
        """Chunk tables using markdown format with smart splitting.
        
        Strategy:
        - Small tables (< chunk_size tokens, single page) → single chunk
        - Large tables → split by rows with headers repeated in each chunk
        - Apply confidence filtering (>0.8) if available
        """
        chunks = []
        chunk_counter = 0
        
        for table in tables:
            # Apply confidence filtering if scores are available
            if table.confidence_scores:
                filtered_table = table.filter_low_confidence_rows(threshold=0.8)
                logger.info(f"Table {table.table_id}: Filtered {len(table.rows) - len(filtered_table.rows)} low-confidence rows")
            else:
                filtered_table = table
            
            # Skip empty tables
            if not filtered_table.rows:
                logger.warning(f"Table {table.table_id} has no rows after filtering, skipping")
                continue
            
            # Get section info
            section_number = filtered_table.section_number or "unknown"
            section_title = filtered_table.section_title or "Unknown Section"
            section_info = hierarchy.get(section_number, {})
            
            if isinstance(section_info, SectionHierarchy):
                section_level = section_info.level
                section_path = section_info.path
                parent_section = section_info.parent
            else:
                section_level = 0
                section_path = [section_number]
                parent_section = None
            
            # Generate markdown for full table
            full_markdown = filtered_table.to_markdown()
            full_tokens = self.count_tokens(full_markdown)
            
            # CASE 1: Small table - single chunk
            if full_tokens <= self.chunk_size and len(filtered_table.page_range) == 1:
                table_metadata = {
                    'table_id': filtered_table.table_id,
                    'table_number': filtered_table.table_number,
                    'table_caption': filtered_table.caption,
                    'is_complete_table': True,
                    'total_rows': len(filtered_table.rows),
                    'num_columns': len(filtered_table.headers) if filtered_table.headers else 0,
                    'confidence_filtered': table.confidence_scores is not None,
                    **filtered_table.to_dict()
                }
                
                # Add acronyms found in table content
                if acronyms:
                    table_text = full_markdown + " " + (filtered_table.caption or "")
                    found_acronyms = self._extract_acronyms_from_text(table_text, acronyms)
                    if found_acronyms:
                        table_metadata['acronyms'] = found_acronyms
                
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{start_id + chunk_counter}",
                    content=full_markdown,
                    chunk_type='table',
                    section_number=section_number,
                    section_title=section_title,
                    section_level=section_level,
                    parent_section=parent_section,
                    section_path=section_path,
                    page_start=filtered_table.page_range[0],
                    page_end=filtered_table.page_range[-1],
                    token_count=full_tokens,
                    metadata=table_metadata
                )
                chunks.append(chunk)
                chunk_counter += 1
                logger.info(f"Table {filtered_table.table_id}: Single chunk ({full_tokens} tokens)")
            
            # CASE 2: Large table - split with headers repeated
            else:
                table_chunks = self._split_large_table(
                    filtered_table,
                    section_number,
                    section_title,
                    section_level,
                    section_path,
                    parent_section,
                    acronyms,
                    start_id + chunk_counter,
                    confidence_filtered=table.confidence_scores is not None
                )
                chunks.extend(table_chunks)
                chunk_counter += len(table_chunks)
                logger.info(f"Table {filtered_table.table_id}: Split into {len(table_chunks)} chunks")
        
        return chunks
    
    def _split_large_table(
        self,
        table: ParsedTable,
        section_number: str,
        section_title: str,
        section_level: int,
        section_path: List[str],
        parent_section: Optional[str],
        acronyms: Optional[Dict[str, AcronymDefinition]],
        start_id: int,
        confidence_filtered: bool
    ) -> List[DocumentChunk]:
        """Split large table into multiple chunks with headers repeated.
        
        Args:
            table: ParsedTable to split
            section_number: Section containing this table
            section_title: Title of the section
            section_level: Level of the section
            section_path: Hierarchical path
            parent_section: Parent section number
            start_id: Starting chunk ID
            confidence_filtered: Whether confidence filtering was applied
        
        Returns:
            List of table chunks with headers in each
        """
        chunks = []
        
        # Calculate optimal rows per chunk
        # Estimate tokens per row
        if table.rows:
            sample_row_markdown = table.to_markdown(row_range=(0, 1))
            sample_tokens = self.count_tokens(sample_row_markdown)
            
            # Reserve space for caption and headers (roughly 100 tokens)
            available_tokens = self.chunk_size - 100
            estimated_tokens_per_row = max(10, (sample_tokens - 100))  # Avoid division by zero
            
            max_rows_per_chunk = max(5, available_tokens // estimated_tokens_per_row)
        else:
            max_rows_per_chunk = 30  # Default
        
        num_rows = len(table.rows)
        total_parts = (num_rows + max_rows_per_chunk - 1) // max_rows_per_chunk
        
        logger.info(f"Splitting table with {num_rows} rows into ~{total_parts} chunks ({max_rows_per_chunk} rows/chunk)")
        
        for i in range(0, num_rows, max_rows_per_chunk):
            row_end = min(i + max_rows_per_chunk, num_rows)
            part_number = i // max_rows_per_chunk + 1
            
            # Generate markdown with headers for this subset
            chunk_markdown = table.to_markdown(
                row_range=(i, row_end),
                include_part_info=True
            )
            
            table_metadata = {
                'table_id': table.table_id,
                'table_number': table.table_number,
                'table_caption': table.caption,
                'is_complete_table': False,
                'chunk_index': part_number - 1,
                'total_chunks': total_parts,
                'row_range': [i, row_end],
                'total_rows': num_rows,
                'num_columns': len(table.headers) if table.headers else 0,
                'confidence_filtered': confidence_filtered,
                **table.to_dict()
            }
            
            # Add acronyms found in table chunk
            if acronyms:
                table_text = chunk_markdown + " " + (table.caption or "")
                found_acronyms = self._extract_acronyms_from_text(table_text, acronyms)
                if found_acronyms:
                    table_metadata['acronyms'] = found_acronyms
            
            chunk = DocumentChunk(
                chunk_id=f"chunk_{start_id + part_number - 1}",
                content=chunk_markdown,
                chunk_type='table',
                section_number=section_number,
                section_title=section_title,
                section_level=section_level,
                parent_section=parent_section,
                section_path=section_path,
                page_start=table.page_range[0],
                page_end=table.page_range[-1],
                token_count=self.count_tokens(chunk_markdown),
                metadata=table_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_diagrams(
        self,
        diagrams: List,
        hierarchy: Dict,
        acronyms: Optional[Dict[str, AcronymDefinition]],
        start_id: int
    ) -> List[DocumentChunk]:
        """Create chunks for diagram references with relationship analysis."""
        chunks = []
        
        for idx, diagram in enumerate(diagrams):
            # Build diagram chunk content focused on relationships and insights
            content_parts = [f"**{diagram.caption}**"]
            
            # Add comprehensive analysis (relationships, insights, context)
            if hasattr(diagram, 'diagram_description') and diagram.diagram_description:
                content_parts.append(f"\n{diagram.diagram_description}")
            
            # Add any extracted data/labels as supplementary info
            if hasattr(diagram, 'extracted_text') and diagram.extracted_text and diagram.extracted_text.strip():
                # Only add if it's different from description (avoid duplication)
                desc = getattr(diagram, 'diagram_description', '')
                if diagram.extracted_text not in desc:
                    content_parts.append(f"\n**Key Elements**: {diagram.extracted_text}")
            
            # Add page reference
            content_parts.append(f"\n*Located on page {diagram.page}*")
            
            diagram_text = "\n".join(content_parts)
            
            # Find which section this diagram belongs to
            section_number = getattr(diagram, 'section_number', None) or "unknown"
            section_title_val = getattr(diagram, 'section_title', None) or "Unknown Section"
            
            section_info = hierarchy.get(section_number, {})
            
            if isinstance(section_info, SectionHierarchy):
                section_title = section_info.title
                section_level = section_info.level
                section_path = section_info.path
                parent_section = section_info.parent
            else:
                section_title = section_title_val
                section_level = 0
                section_path = [section_number]
                parent_section = None
            
            chunk = DocumentChunk(
                chunk_id=f"chunk_{start_id + idx}",
                content=diagram_text,
                chunk_type='diagram_ref',
                section_number=section_number,
                section_title=section_title,
                section_level=section_level,
                parent_section=parent_section,
                section_path=section_path,
                page_start=diagram.page,
                page_end=diagram.page,
                token_count=self.count_tokens(diagram_text),
                metadata={
                    'diagram_id': diagram.diagram_id,
                    'diagram_caption': diagram.caption,
                    'figure_number': getattr(diagram, 'figure_number', None),
                    'image_path': getattr(diagram, 'image_path', None),
                    'has_relationship_analysis': bool(getattr(diagram, 'diagram_description', None)),
                    'has_extracted_data': bool(getattr(diagram, 'extracted_text', None)),
                }
            )
            
            # Add acronyms found in diagram caption and description
            if acronyms:
                found_acronyms = self._extract_acronyms_from_text(diagram_text, acronyms)
                if found_acronyms:
                    chunk.metadata['acronyms'] = found_acronyms
            
            chunks.append(chunk)
        
        return chunks
    
    def _get_section_cross_refs(self, section_number: str, cross_refs: List) -> Dict:
        """Get cross-references for a section."""
        outgoing = []
        incoming = []
        
        for ref in cross_refs:
            if hasattr(ref, 'source_section') and ref.source_section == section_number:
                outgoing.append(ref.target_section)
            if hasattr(ref, 'target_section') and ref.target_section == section_number:
                incoming.append(ref.source_section)
        
        return {
            'outgoing': list(set(outgoing)),
            'incoming': list(set(incoming))
        }
    
    def save_chunks(self, chunks: List[DocumentChunk], output_path: str):
        """
        Save chunks to JSON file.
        
        Args:
            chunks: List of document chunks
            output_path: Path to save JSON file
        """
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        chunks_data = [chunk.to_dict() for chunk in chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    """Main function for testing."""
    import json
    from src.pdf_parser import ParsedDocument, ParsedSection, ParsedTable
    from src.metadata_extractor import MetadataExtractor
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = get_config()
    
    # Load parsed document
    parsed_output = config.get('paths.parsed_output')
    with open(parsed_output, 'r') as f:
        parsed_data = json.load(f)
    
    # Reconstruct ParsedDocument
    sections = [ParsedSection(**s) for s in parsed_data.get('sections', [])]
    tables = [ParsedTable(**t) for t in parsed_data.get('tables', [])]
    
    parsed_doc = ParsedDocument(
        title=parsed_data['title'],
        sections=sections,
        tables=tables,
        diagrams=[],
        metadata=parsed_data['metadata'],
        raw_text=parsed_data.get('raw_text', '')
    )
    
    # Extract metadata
    extractor = MetadataExtractor(config)
    metadata = extractor.extract_all_metadata(parsed_doc)
    
    # Chunk document
    chunker = HierarchicalChunker(config)
    chunks = chunker.chunk_document(parsed_doc, metadata, metadata.get('acronyms'))
    
    # Save chunks
    chunks_output = config.get('paths.chunks_output')
    chunker.save_chunks(chunks, chunks_output)
    
    print(f"\n✓ Chunking complete!")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Text chunks: {sum(1 for c in chunks if c.chunk_type == 'text')}")
    print(f"  Table chunks: {sum(1 for c in chunks if c.chunk_type == 'table')}")
    print(f"  Diagram chunks: {sum(1 for c in chunks if c.chunk_type == 'diagram_ref')}")
    print(f"  Output: {chunks_output}")


if __name__ == '__main__':
    main()
