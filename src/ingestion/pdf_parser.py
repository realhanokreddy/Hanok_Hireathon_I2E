"""
PDF Parser module using Docling for advanced structure-preserving PDF parsing.
Handles tables, diagrams, section hierarchy, and cross-references.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests
from tqdm import tqdm

# Docling imports
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    logging.warning("Docling not installed. Install with: pip install docling")

# Fallback imports
try:
    import pymupdf as fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Google Gemini imports
try:
    import google.generativeai as genai
    from PIL import Image
    import io
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logging.warning("Google Generative AI not installed. Install with: pip install google-generativeai Pillow")

from src.config import get_config


logger = logging.getLogger(__name__)


@dataclass
class ParsedSection:
    """Represents a parsed section from the document."""
    section_number: str
    title: str
    content: str
    page_start: int
    page_end: int
    level: int
    parent_section: Optional[str] = None
    subsections: List[str] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []


@dataclass
class ParsedTable:
    """Represents a parsed table from the document."""
    table_id: str
    caption: str
    page_range: List[int]  # Support multi-page tables: [45] or [45, 46, 47]
    headers: List[str]
    rows: List[List[str]]
    
    # Enhanced metadata
    table_number: Optional[str] = None  # e.g., "4.1", "A-2"
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    confidence_scores: Optional[Dict] = None  # (row, col) -> confidence float
    
    def to_markdown(self, row_range: Optional[tuple] = None, include_part_info: bool = False) -> str:
        """Convert table to markdown format.
        
        Args:
            row_range: Optional (start, end) tuple to render subset of rows
            include_part_info: Whether to add part info for chunked tables
        
        Returns:
            Markdown formatted table
        """
        lines = []
        
        # Caption with table number
        caption_text = f"**Table {self.table_number}: {self.caption}**" if self.table_number else f"**{self.caption}**"
        
        # Add part info if this is a chunk
        if include_part_info and row_range:
            caption_text += f" (Rows {row_range[0]+1}-{row_range[1]})"
        
        lines.append(caption_text)
        lines.append("")  # Empty line
        
        # Headers — prepend a row-index column "#"
        if self.headers:
            lines.append("| # | " + " | ".join(self.headers) + " |")
            lines.append("|" + "|".join(["---"] * (len(self.headers) + 1)) + "|")

        # Rows (full or subset) — row numbers are 1-based and absolute
        row_start = row_range[0] if row_range else 0
        rows_to_render = self.rows[row_range[0]:row_range[1]] if row_range else self.rows
        for i, row in enumerate(rows_to_render):
            row_num = row_start + i + 1
            lines.append("| " + str(row_num) + " | " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(lines)
    
    def to_text(self) -> str:
        """Legacy text representation (deprecated - use to_markdown instead)."""
        return self.to_markdown()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for metadata."""
        return {
            'table_id': self.table_id,
            'table_number': self.table_number,
            'caption': self.caption,
            'pages': self.page_range,
            'section_number': self.section_number,
            'section_title': self.section_title,
            'num_rows': len(self.rows),
            'num_columns': len(self.headers) if self.headers else 0,
            'headers': self.headers
        }
    
    def get_high_confidence_rows(self, threshold: float = 0.8) -> List[int]:
        """Get row indices with confidence scores above threshold.
        
        Args:
            threshold: Minimum confidence score (default 0.8)
        
        Returns:
            List of row indices passing the threshold
        """
        if not self.confidence_scores:
            # If no confidence scores, assume all rows are good
            return list(range(len(self.rows)))
        
        high_confidence_rows = set()
        for (row_idx, col_idx), score in self.confidence_scores.items():
            if score >= threshold:
                high_confidence_rows.add(row_idx)
        
        return sorted(list(high_confidence_rows))
    
    def filter_low_confidence_rows(self, threshold: float = 0.8) -> 'ParsedTable':
        """Create new table with only high-confidence rows.
        
        Args:
            threshold: Minimum confidence score
        
        Returns:
            New ParsedTable with filtered rows
        """
        high_conf_indices = self.get_high_confidence_rows(threshold)
        
        if len(high_conf_indices) == len(self.rows):
            return self  # All rows pass
        
        filtered_rows = [self.rows[i] for i in high_conf_indices]
        
        return ParsedTable(
            table_id=self.table_id,
            caption=self.caption,
            page_range=self.page_range,
            headers=self.headers,
            rows=filtered_rows,
            table_number=self.table_number,
            section_number=self.section_number,
            section_title=self.section_title,
            confidence_scores=self.confidence_scores
        )


@dataclass
class ParsedDiagram:
    """Represents a diagram/figure from the document with extracted information."""
    diagram_id: str
    caption: str
    page: int
    figure_number: Optional[str] = None  # Extracted figure number like "2.5-1"
    image_path: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    extracted_text: Optional[str] = None  # Text extracted from image via Gemini
    diagram_description: Optional[str] = None  # Description of the diagram
    image_size_bytes: Optional[int] = None  # Size for filtering


@dataclass
class ParsedDocument:
    """Complete parsed document structure."""
    title: str
    sections: List[ParsedSection]
    tables: List[ParsedTable]
    diagrams: List[ParsedDiagram]
    metadata: Dict[str, Any]
    raw_text: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'sections': [asdict(s) for s in self.sections],
            'tables': [asdict(t) for t in self.tables],
            'diagrams': [asdict(d) for d in self.diagrams],
            'metadata': self.metadata,
            'raw_text': self.raw_text
        }


class PDFParser:
    """PDF parser with Docling as primary and fallbacks."""
    
    def __init__(self, config=None):
        """
        Initialize PDF parser.
        
        Args:
            config: Configuration instance (uses global config if None)
        """
        self.config = config or get_config()
        self.parser_config = self.config.get_pdf_parser_config()
        self.primary_parser = self.parser_config.get('primary', 'docling')
        self.fallback_parser = self.parser_config.get('fallback', 'pymupdf')
        
        # Flag set to True when Gemini hits a rate limit — remaining images skip analysis
        # gracefully rather than terminating ingestion
        self._gemini_rate_limited = False
        self._gemini_analyzed_count = 0
        self._gemini_skipped_count = 0
        
        logger.info(f"Initialized PDF parser: primary={self.primary_parser}, fallback={self.fallback_parser}")
    
    def download_pdf(self, url: str, output_path: str) -> str:
        """
        Download PDF from URL.
        
        Args:
            url: URL to download from
            output_path: Path to save the PDF
            
        Returns:
            Path to downloaded file
        """
        logger.info(f"Downloading PDF from {url}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        download_config = self.config.get('download', {})
        timeout = download_config.get('timeout', 300)
        chunk_size = download_config.get('chunk_size', 8192)
        
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"PDF downloaded to {output_path}")
        return str(output_path)
    
    def parse_with_docling(self, pdf_path: str) -> ParsedDocument:
        """
        Parse PDF using Docling in text-only mode for memory efficiency.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ParsedDocument instance
        """
        if not HAS_DOCLING:
            raise ImportError("Docling not installed. Install with: pip install docling")
        
        logger.info(f"Parsing PDF with Docling (text-only mode): {pdf_path}")
        
        # Configure Docling pipeline options for text-only processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Disable OCR to save memory
        pipeline_options.do_table_structure = self.parser_config.get('extract_tables', True)
        pipeline_options.images_scale = 1.0  # Don't scale images (save memory)
        pipeline_options.generate_page_images = False  # Skip page image generation
        pipeline_options.generate_picture_images = False  # Skip picture extraction
        
        # Create converter with text-only backend
        try:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=PyPdfiumDocumentBackend  # Lightweight text backend
                    )
                }
            )
        except Exception as e:
            logger.warning(f"Error creating converter with backend: {e}, trying without backend")
            # Fallback to basic converter
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        
        # Convert document
        logger.info("Converting PDF (this may take 2-3 minutes for large documents)...")
        result = converter.convert(pdf_path)
        doc = result.document
        
        # Extract sections
        sections = self._extract_sections_from_docling(doc)
        
        # Extract tables
        tables = self._extract_tables_from_docling(doc)
        
        # Extract diagrams/figures
        diagrams = self._extract_diagrams_from_docling(doc)
        
        # Assign section context to tables and diagrams
        self._assign_sections_to_tables(tables, sections)
        self._assign_sections_to_diagrams(diagrams, sections)
        
        # Get full text
        raw_text = doc.export_to_markdown()
        
        # Build metadata
        metadata = {
            'source': pdf_path,
            'parser': 'docling',
            'num_pages': len(doc.pages) if hasattr(doc, 'pages') else 0,
            'num_sections': len(sections),
            'num_tables': len(tables),
            'num_diagrams': len(diagrams)
        }
        
        parsed_doc = ParsedDocument(
            title=self._extract_title(doc),
            sections=sections,
            tables=tables,
            diagrams=diagrams,
            metadata=metadata,
            raw_text=raw_text
        )
        
        logger.info(f"Parsing complete: {len(sections)} sections, {len(tables)} tables, {len(diagrams)} diagrams")
        if self._gemini_analyzed_count > 0 or self._gemini_skipped_count > 0:
            logger.info(
                f"Gemini image analysis: {self._gemini_analyzed_count} analyzed, "
                f"{self._gemini_skipped_count} skipped (rate-limited)"
            )
        return parsed_doc
    
    def _extract_sections_from_docling(self, doc) -> List[ParsedSection]:
        """Extract hierarchical sections from Docling document."""
        sections = []
        section_counter = {}
        
        # Iterate through document items
        for item in doc.body.iterate_items():
            # Check if it's a heading/section
            if hasattr(item, 'label') and 'section' in item.label.lower():
                level = self._determine_section_level(item)
                section_num = self._get_section_number(item, level, section_counter)
                
                section = ParsedSection(
                    section_number=section_num,
                    title=item.text if hasattr(item, 'text') else '',
                    content='',  # Will be filled by aggregating following content
                    page_start=item.prov[0].page if hasattr(item, 'prov') and item.prov else 0,
                    page_end=item.prov[0].page if hasattr(item, 'prov') and item.prov else 0,
                    level=level,
                    parent_section=self._find_parent_section(section_num, sections)
                )
                sections.append(section)
        
        # Fill section content
        self._fill_section_content(doc, sections)
        
        return sections
    
    def _extract_tables_from_docling(self, doc) -> List[ParsedTable]:
        """Extract tables from Docling document with enhanced metadata."""
        tables = []
        table_id = 0
        
        for item in doc.body.iterate_items():
            if hasattr(item, 'label') and item.label == 'table':
                table_id += 1
                
                # Extract table data
                headers = []
                rows = []
                confidence_scores = {}
                
                if hasattr(item, 'data') and item.data:
                    table_data = item.data
                    if hasattr(table_data, 'grid'):
                        # Extract from grid structure
                        grid = table_data.grid
                        if len(grid) > 0:
                            headers = grid[0] if grid else []
                            rows = grid[1:] if len(grid) > 1 else []
                    
                    # Extract confidence scores if available
                    if hasattr(table_data, 'confidence') and table_data.confidence:
                        confidence_scores = table_data.confidence
                
                # Extract caption and table number
                caption = item.text if hasattr(item, 'text') else f"Table {table_id}"
                table_number = self._extract_table_number(caption)
                
                # Get page range (support multi-page tables)
                if hasattr(item, 'prov') and item.prov:
                    pages = sorted(list(set(p.page for p in item.prov)))
                else:
                    pages = [0]
                
                # Validate rows: check if all rows have same column count
                if headers and rows:
                    expected_cols = len(headers)
                    rows = [row for row in rows if len(row) == expected_cols]
                
                table = ParsedTable(
                    table_id=f"table_{table_id}",
                    caption=caption,
                    page_range=pages,
                    headers=headers,
                    rows=rows,
                    table_number=table_number,
                    confidence_scores=confidence_scores if confidence_scores else None
                )
                tables.append(table)
        
        return tables
    
    def _extract_table_number(self, caption: str) -> Optional[str]:
        """Extract table number from caption.
        
        Supports formats like:
        - "Table 4.1: Mission Phases"
        - "Table A-2 - Cost Breakdown"
        - "4.1 Mission Phases"
        
        Args:
            caption: Table caption text
        
        Returns:
            Extracted table number or None
        """
        import re
        
        # Pattern 1: "Table X.Y" or "Table X-Y" or "Table X"
        match = re.search(r'[Tt]able\s+([A-Z0-9]+[-.]?[0-9]*)', caption)
        if match:
            return match.group(1)
        
        # Pattern 2: Starting with number "4.1 Something"
        match = re.match(r'^([0-9]+\.[0-9]+)', caption)
        if match:
            return match.group(1)
        
        # Pattern 3: "Appendix X Table Y"
        match = re.search(r'[Aa]ppendix\s+([A-Z])\s+[Tt]able\s+([0-9]+)', caption)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        return None
    
    def _analyze_image_with_gemini(self, image_bytes: bytes, caption: str = "") -> Dict[str, str]:
        """
        Analyze image using Google Gemini to extract text and description.
        
        Image analysis is optional — if Gemini is unavailable, rate-limited, or
        fails for any image, ingestion continues and the diagram is still created
        with just its caption.
        
        Args:
            image_bytes: Image data in bytes
            caption: Figure caption for context
            
        Returns:
            Dictionary with 'extracted_text' and 'description'
        """
        if not HAS_GEMINI:
            logger.warning("Gemini not available. Skipping image analysis.")
            return {"extracted_text": "", "description": ""}
        
        # Once rate-limited, skip all further Gemini calls for this session
        if self._gemini_rate_limited:
            self._gemini_skipped_count += 1
            return {"extracted_text": "", "description": ""}
        
        try:
            # Initialize Gemini
            gemini_config = self.config.get('gemini', {})
            api_key = gemini_config.get('api_key')
            
            if not api_key:
                logger.warning("Gemini API key not configured. Set GEMINI_API_KEY in env/.env")
                return {"extracted_text": "", "description": ""}
            
            genai.configure(api_key=api_key)
            model_name = gemini_config.get('model', 'models/gemini-2.5-flash')
            model = genai.GenerativeModel(model_name)
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Prompt for deep analysis of relationships and insights
            prompt = f"""You are analyzing a technical diagram/chart from NASA's Systems Engineering Handbook.
Figure caption: "{caption}"

Analyze this visualization and provide a comprehensive understanding:

1. **Type of Visualization**: Identify what kind of diagram/chart this is (pie chart, bar graph, flowchart, process diagram, organizational chart, timeline, architecture diagram, network diagram, etc.)

2. **Data & Relationships**: 
   - What data or information is being presented?
   - What relationships, connections, or flows are shown between elements?
   - Are there hierarchies, dependencies, or sequences?
   - What comparisons or proportions are illustrated?

3. **Key Insights**:
   - What is the main message or purpose of this visualization?
   - What patterns, trends, or important points does it convey?
   - Are there critical values, percentages, or measurements shown?

4. **Context & Meaning**:
   - How do the elements relate to each other?
   - What process, system, or concept is being explained?
   - Are there cause-and-effect relationships or workflows?

Format your response as:
VISUALIZATION_TYPE:
[type of chart/diagram]

RELATIONSHIPS_AND_DATA:
[detailed explanation of relationships, connections, data flows, hierarchies, comparisons, etc.]

KEY_INSIGHTS:
[main message, patterns, critical values, what the visualization is trying to communicate]

CONTEXT:
[how elements relate, what system/process is shown, overall meaning]"""
            
            # Call Gemini
            response = model.generate_content([prompt, image])
            
            # Parse response
            result_text = response.text
            
            # Extract sections from response
            viz_type = ""
            relationships = ""
            insights = ""
            context = ""
            
            # Parse structured response
            if "VISUALIZATION_TYPE:" in result_text:
                parts = result_text.split("RELATIONSHIPS_AND_DATA:")
                viz_type = parts[0].replace("VISUALIZATION_TYPE:", "").strip()
                
                if len(parts) > 1:
                    rest = parts[1]
                    if "KEY_INSIGHTS:" in rest:
                        rel_parts = rest.split("KEY_INSIGHTS:")
                        relationships = rel_parts[0].strip()
                        
                        if len(rel_parts) > 1:
                            insights_rest = rel_parts[1]
                            if "CONTEXT:" in insights_rest:
                                insight_parts = insights_rest.split("CONTEXT:")
                                insights = insight_parts[0].strip()
                                if len(insight_parts) > 1:
                                    context = insight_parts[1].strip()
                            else:
                                insights = insights_rest.strip()
            
            # Build comprehensive description
            description_parts = []
            if viz_type:
                description_parts.append(f"**Type**: {viz_type}")
            if relationships:
                description_parts.append(f"**Relationships**: {relationships}")
            if insights:
                description_parts.append(f"**Key Insights**: {insights}")
            if context:
                description_parts.append(f"**Context**: {context}")
            
            description = "\n\n".join(description_parts) if description_parts else result_text.strip()
            
            # For backward compatibility, extract any visible text/labels mentioned
            extracted_text = ""
            if relationships:
                # Try to extract any specific labels/values mentioned
                extracted_text = relationships[:500]  # Use relationship info as extracted text
            
            self._gemini_analyzed_count += 1
            logger.info(f"Successfully analyzed image with Gemini [{self._gemini_analyzed_count}] (caption: {caption[:50]}...)")
            
            return {
                "extracted_text": extracted_text,
                "description": description
            }
            
        except Exception as e:
            error_str = str(e).lower()
            # Detect rate limit / quota / timeout errors — stop attempting further Gemini calls
            if any(kw in error_str for kw in ['429', 'rate limit', 'quota', 'resourceexhausted', 'resource_exhausted', 'too many requests', '504', 'deadline exceeded', 'timeout', 'timed out', 'unavailable', '503']):
                self._gemini_rate_limited = True
                logger.warning(
                    f"Gemini rate limit hit after {self._gemini_analyzed_count} images analyzed. "
                    f"Remaining images will be stored with caption only — ingestion continues."
                )
            else:
                logger.warning(f"Error analyzing image with Gemini (skipping this image): {e}")
            return {"extracted_text": "", "description": ""}
    
    def _extract_images_from_pdf(self, pdf_doc, page_num: int, sections: List[ParsedSection], seen_xrefs: set = None) -> List[ParsedDiagram]:
        """
        Extract images from a PDF page using PyMuPDF.
        
        Args:
            pdf_doc: PyMuPDF document object
            page_num: Page number (0-indexed)
            sections: List of sections for context assignment
            seen_xrefs: Set of already-processed xrefs to avoid duplicates across pages
            
        Returns:
            List of ParsedDiagram objects
        """
        if seen_xrefs is None:
            seen_xrefs = set()
        diagrams = []
        min_size = self.parser_config.get('min_image_size', 10000)
        min_dim = self.parser_config.get('min_image_dimension', 100)  # pixels
        analyze_with_gemini = self.parser_config.get('analyze_images_with_gemini', True)
        
        try:
            page = pdf_doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]

                    # Skip images already processed on another page (e.g. logos, headers)
                    if xref in seen_xrefs:
                        continue

                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    img_width = base_image.get("width", 0)
                    img_height = base_image.get("height", 0)
                    
                    # Filter small images (icons, logos, etc.) by byte size
                    if len(image_bytes) < min_size:
                        continue

                    # Filter images that are too small in pixel dimensions
                    if img_width < min_dim or img_height < min_dim:
                        continue

                    # Only keep images with a real figure caption nearby
                    caption = self._find_image_caption(page, img)
                    if not caption:
                        continue

                    seen_xrefs.add(xref)
                    
                    diagram_id = f"diagram_p{page_num+1}_i{img_index+1}"
                    
                    # Analyze with Gemini if enabled
                    extracted_text = ""
                    description = ""
                    
                    if analyze_with_gemini:
                        if self._gemini_rate_limited:
                            logger.debug(f"Skipping Gemini analysis for {diagram_id} (rate-limited)")
                        else:
                            result = self._analyze_image_with_gemini(image_bytes, caption)
                            extracted_text = result["extracted_text"]
                            description = result["description"]
                    
                    # Assign section context
                    section_number = None
                    section_title = None
                    for section in sections:
                        if section.page_start <= page_num + 1 <= section.page_end:
                            section_number = section.section_number
                            section_title = section.title
                            break
                    
                    # Extract figure number from caption
                    figure_number = self._extract_figure_number(caption)
                    
                    diagram = ParsedDiagram(
                        diagram_id=diagram_id,
                        caption=caption,
                        page=page_num + 1,
                        figure_number=figure_number,
                        image_path=None,
                        section_number=section_number,
                        section_title=section_title,
                        extracted_text=extracted_text,
                        diagram_description=description,
                        image_size_bytes=len(image_bytes)
                    )
                    
                    diagrams.append(diagram)
                    logger.info(f"Extracted diagram: {diagram_id} from page {page_num+1}")
                    
                except Exception as e:
                    logger.warning(f"Error extracting image {img_index} from page {page_num+1}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error extracting images from page {page_num+1}: {e}")
        
        return diagrams
    
    def _find_image_caption(self, page, img) -> str:
        """Try to find caption text near an image, handling various formats like 'FIGURE 2.5-1 Description'."""
        try:
            # Get image position - img is tuple from get_images()
            # Format: (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, bbox)
            try:
                img_rect = page.get_image_bbox(img)
            except:
                # If bbox not directly available, try to find it differently
                xref = img[0]
                # Search all image instances on the page
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    return ""
                img_rect = img_rects[0]  # Use first instance
            
            page_rect = page.rect
            import re as _re

            # Regex: matches "FIGURE 6.7-2 ..." anywhere in extracted text
            _CAPTION_RE = _re.compile(
                r'((?:FIGURE|FIG\.?|DIAGRAM)\s+[\d][\d\.\-]*\b.*)',
                _re.IGNORECASE
            )

            def _extract_caption(raw: str) -> str:
                """Return the FIGURE/DIAGRAM caption line from raw textbox output."""
                cleaned = ' '.join(raw.split())
                m = _CAPTION_RE.search(cleaned)
                return m.group(1).strip() if m else ""

            # Strategy 1: Search for "FIGURE" / "Fig." text on the page and find
            #             the instance closest to this image (within 200 pt).
            figure_instances = page.search_for("FIGURE", flags=fitz.TEXT_DEHYPHENATE)
            figure_instances.extend(page.search_for("Figure", flags=fitz.TEXT_DEHYPHENATE))
            figure_instances.extend(page.search_for("Fig.", flags=fitz.TEXT_DEHYPHENATE))

            best_caption = ""
            min_distance = float('inf')

            for fig_rect in figure_instances:
                if fig_rect.y0 >= img_rect.y1:      # below image
                    distance = fig_rect.y0 - img_rect.y1
                elif fig_rect.y1 <= img_rect.y0:    # above image
                    distance = img_rect.y0 - fig_rect.y1
                else:                               # overlapping
                    distance = 0

                if distance < 200 and distance < min_distance:
                    # Anchor the rect at the left edge of the found "FIGURE" word
                    # so that get_textbox starts exactly at the caption, not earlier.
                    caption_rect = fitz.Rect(
                        fig_rect.x0,
                        fig_rect.y0,
                        page_rect.x1 - 30,
                        fig_rect.y1 + 120        # room for multi-line captions
                    )
                    raw = page.get_textbox(caption_rect).strip()
                    caption = _extract_caption(raw)
                    if caption:
                        min_distance = distance
                        best_caption = caption

            if best_caption:
                return best_caption

            # Strategy 2: Search text in a band below the image
            search_rect = fitz.Rect(
                page_rect.x0 + 30,
                img_rect.y1,
                page_rect.x1 - 30,
                img_rect.y1 + 120
            )
            caption = _extract_caption(page.get_textbox(search_rect))
            if caption:
                return caption

            # Strategy 3: Search text in a band above the image
            search_rect_above = fitz.Rect(
                page_rect.x0 + 30,
                max(img_rect.y0 - 120, 0),
                page_rect.x1 - 30,
                img_rect.y0
            )
            caption = _extract_caption(page.get_textbox(search_rect_above))
            if caption:
                return caption

            # No caption found near this image
            return ""
            
        except Exception as e:
            logger.warning(f"Error finding caption for image: {e}")
            return ""
    
    def _extract_figure_number(self, caption: str) -> Optional[str]:
        """
        Extract figure number from caption.
        Examples:
        - "FIGURE 2.5-1 Life-Cycle Cost" -> "2.5-1"
        - "Figure 4.1: System Overview" -> "4.1"
        - "Fig. 3-2 Process Flow" -> "3-2"
        """
        import re
        
        # Pattern to match figure numbers like: 2.5-1, 4.1, 3-2, A-1, etc.
        patterns = [
            r'(?:FIGURE|Figure|Fig\.?)\s+([\d]+\.[\d]+-[\d]+)',  # 2.5-1 format
            r'(?:FIGURE|Figure|Fig\.?)\s+([\d]+\.[\d]+)',        # 4.1 format
            r'(?:FIGURE|Figure|Fig\.?)\s+([\d]+-[\d]+)',         # 3-2 format
            r'(?:FIGURE|Figure|Fig\.?)\s+([A-Z]-[\d]+)',         # A-1 format
            r'(?:FIGURE|Figure|Fig\.?)\s+([\d]+)',               # 4 format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_diagrams_from_docling(self, doc) -> List[ParsedDiagram]:
        """Extract diagram references from Docling document."""
        diagrams = []
        diagram_id = 0
        
        for item in doc.body.iterate_items():
            if hasattr(item, 'label') and item.label in ['figure', 'picture']:
                diagram_id += 1
                
                caption = item.text if hasattr(item, 'text') else f"Figure {diagram_id}"
                figure_number = self._extract_figure_number(caption)
                
                diagram = ParsedDiagram(
                    diagram_id=f"diagram_{diagram_id}",
                    caption=caption,
                    page=item.prov[0].page if hasattr(item, 'prov') and item.prov else 0,
                    figure_number=figure_number,
                    image_path=None  # Could extract image if needed
                )
                diagrams.append(diagram)
        
        return diagrams
    
    def _extract_title(self, doc) -> str:
        """Extract document title."""
        # Try to get from metadata
        if hasattr(doc, 'name'):
            return doc.name
        
        # Try to find first heading
        for item in doc.body.iterate_items():
            if hasattr(item, 'label') and 'title' in item.label.lower():
                return item.text
        
        return "NASA Systems Engineering Handbook"
    
    def _determine_section_level(self, item) -> int:
        """Determine section level from heading style."""
        # Docling provides level information
        if hasattr(item, 'level'):
            return item.level
        
        # Fallback: analyze text style
        if hasattr(item, 'label'):
            label = item.label.lower()
            if 'h1' in label or 'title' in label:
                return 1
            elif 'h2' in label:
                return 2
            elif 'h3' in label:
                return 3
            elif 'h4' in label:
                return 4
        
        return 1
    
    def _get_section_number(self, item, level: int, counter: Dict) -> str:
        """Generate section number."""
        # Try to extract from text
        if hasattr(item, 'text'):
            import re
            match = re.match(r'^(\d+(?:\.\d+)*)', item.text)
            if match:
                return match.group(1)
        
        # Generate based on level and counter
        if level not in counter:
            counter[level] = 0
        counter[level] += 1
        
        # Reset lower level counters
        for l in range(level + 1, 10):
            if l in counter:
                del counter[l]
        
        # Build section number
        parts = [str(counter.get(l, 1)) for l in range(1, level + 1) if l in counter]
        return '.'.join(parts)
    
    def _find_parent_section(self, section_num: str, sections: List[ParsedSection]) -> Optional[str]:
        """Find parent section number."""
        parts = section_num.split('.')
        if len(parts) <= 1:
            return None
        
        parent_num = '.'.join(parts[:-1])
        return parent_num
    
    def _fill_section_content(self, doc, sections: List[ParsedSection]):
        """Fill content for each section by aggregating following text."""
        # This is a simplified version - would need more sophisticated logic
        # to properly associate content with sections
        for section in sections:
            section.content = f"Content for section {section.section_number}"
    
    def _assign_sections_to_tables(self, tables: List[ParsedTable], sections: List[ParsedSection]) -> None:
        """Assign section context to tables based on page numbers.
        
        Args:
            tables: List of parsed tables
            sections: List of parsed sections
        """
        for table in tables:
            # Find the section that contains this table's first page
            table_start_page = table.page_range[0] if table.page_range else 0
            
            best_section = None
            for section in sections:
                # Check if table is within section's page range
                if section.page_start <= table_start_page <= section.page_end:
                    # Prefer most specific (deepest) section
                    if best_section is None or section.level > best_section.level:
                        best_section = section
            
            if best_section:
                table.section_number = best_section.section_number
                table.section_title = best_section.title
            else:
                # Fallback: find closest preceding section
                for section in reversed(sections):
                    if section.page_end <= table_start_page:
                        table.section_number = section.section_number
                        table.section_title = section.title
                        break
    
    def _assign_sections_to_diagrams(self, diagrams: List[ParsedDiagram], sections: List[ParsedSection]) -> None:
        """Assign section context to diagrams based on page numbers.
        
        Args:
            diagrams: List of parsed diagrams
            sections: List of parsed sections
        """
        for diagram in diagrams:
            diagram_page = diagram.page
            
            best_section = None
            for section in sections:
                # Check if diagram is within section's page range
                if section.page_start <= diagram_page <= section.page_end:
                    # Prefer most specific (deepest) section
                    if best_section is None or section.level > best_section.level:
                        best_section = section
            
            if best_section:
                diagram.section = best_section.section_number
    
          
    
    def parse_with_pymupdf(self, pdf_path: str) -> ParsedDocument:
        """
        Parse the entire PDF with PyMuPDF in a single pass.

        One loop over all pages collects:
          - full document text (concatenated into one string)
          - per-page text with page-number tracking
          - tables (per-page, then merged for multi-page continuations)
          - image xrefs (deduplicated document-wide)

        Section extraction then works on the FULL text string so section
        content naturally spans page breaks without any splitting artefacts.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ParsedDocument instance
        """
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")

        logger.info(f"Parsing PDF with PyMuPDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        logger.info(f"Document has {num_pages} pages — starting single-pass extraction ...")

        full_text = ""          # entire document as one string
        page_offsets = []       # char offset where each page starts in full_text
        page_texts = []         # (page_1idx, plain_text) for TOC detection
        tables: List[ParsedTable] = []
        table_counter = 0
        image_xrefs: dict = {}  # xref -> page_1idx (first page that references it)

        # ── Single pass: text + tables + image xrefs ─────────────────────
        for page_num, page in enumerate(doc):
            page_1idx = page_num + 1
            pt = page.get_text()

            page_offsets.append(len(full_text))
            full_text += pt
            page_texts.append((page_1idx, pt))

            # Tables
            if self.parser_config.get('extract_tables', True):
                try:
                    for table in page.find_tables():
                        table_counter += 1
                        table_data = table.extract()
                        if not table_data or len(table_data) < 2:
                            continue
                        headers = [str(c) if c else "" for c in table_data[0]]
                        rows = [[str(c) if c else "" for c in row] for row in table_data[1:]]
                        rows = [r for r in rows if any(c.strip() for c in r)]
                        if not rows:
                            continue
                        bbox = table.bbox
                        cap_text = page.get_textbox(
                            fitz.Rect(bbox[0], max(0, bbox[1] - 30), bbox[2], bbox[1])
                        )
                        caption = cap_text.strip() if cap_text else f"Table {table_counter}"
                        tables.append(ParsedTable(
                            table_id=f"table_{table_counter}",
                            caption=caption,
                            page_range=[page_1idx],
                            headers=headers,
                            rows=rows,
                            table_number=self._extract_table_number(caption),
                            section_number=None,
                            section_title=None,
                            confidence_scores=None,
                        ))
                except Exception as e:
                    logger.warning(f"Error extracting tables from page {page_1idx}: {e}")

            # Collect unique image xrefs (first page wins)
            if self.parser_config.get('extract_images', False):
                for img in page.get_images(full=True):
                    xref = img[0]
                    if xref not in image_xrefs:
                        image_xrefs[xref] = page_1idx

        logger.info(
            f"Pass complete — {len(full_text):,} chars of text, "
            f"{table_counter} raw table fragments, "
            f"{len(image_xrefs)} unique image xrefs"
        )

        # ── Sections: regex on full text, content spans page breaks ──────
        sections = self._extract_sections_from_full_text(
            full_text, page_offsets, page_texts
        )

        # ── Tables: merge multi-page fragments ───────────────────────────
        tables = self._merge_multipage_tables(tables)
        logger.info(f"Tables after merge: {len(tables)}")

        # ── Diagrams: caption-based rendering (handles vector figures) ───
        diagrams: List[ParsedDiagram] = []
        if self.parser_config.get('extract_images', False):
            toc_pages = {pn for pn, txt in page_texts if self._is_toc_page(txt)}
            logger.info("Extracting diagrams by caption-based page rendering ...")
            diagrams = self._extract_diagrams_by_caption_rendering(doc, sections, toc_pages)
            logger.info(f"Extracted {len(diagrams)} diagrams")

        # ── Assign sections to tables ─────────────────────────────────────
        if sections and tables:
            self._assign_sections_to_tables(tables, sections)

        metadata = {
            'source': pdf_path,
            'parser': 'pymupdf',
            'num_pages': num_pages,
            'num_sections': len(sections),
            'num_tables': len(tables),
            'num_diagrams': len(diagrams),
        }
        doc.close()

        logger.info(
            f"PyMuPDF parsing complete: {len(sections)} sections, "
            f"{len(tables)} tables, {len(diagrams)} diagrams"
        )
        return ParsedDocument(
            title="NASA Systems Engineering Handbook",
            sections=sections,
            tables=tables,
            diagrams=diagrams,
            metadata=metadata,
            raw_text=full_text,
        )

    def _extract_sections_from_full_text(
        self,
        full_text: str,
        page_offsets: List[int],
        page_texts: List[tuple],
    ) -> List[ParsedSection]:
        """
        Extract sections by running a single regex over the entire document text.

        Because full_text is one continuous string, section content naturally
        includes text that spans page breaks — no fragmentation artefacts.

        Page numbers are recovered by binary-searching page_offsets for the
        character position of each heading match.
        """
        import re
        import bisect

        # Detect and skip TOC pages
        toc_pages = {pn for pn, txt in page_texts if self._is_toc_page(txt)}
        if toc_pages:
            logger.info(f"TOC pages detected (will be excluded from content): {sorted(toc_pages)}")

        # Build a set of char ranges to skip (text of TOC pages)
        skip_ranges: List[tuple] = []
        for pn, pt in page_texts:
            if pn in toc_pages:
                start = page_offsets[pn - 1]
                end = page_offsets[pn] if pn < len(page_offsets) else len(full_text)
                skip_ranges.append((start, end))

        def in_skip_range(pos: int) -> bool:
            for s, e in skip_ranges:
                if s <= pos < e:
                    return True
            return False

        def char_pos_to_page(pos: int) -> int:
            """Return 1-indexed page number for a char position in full_text."""
            idx = bisect.bisect_right(page_offsets, pos) - 1
            return max(1, idx + 1)

        # Section heading pattern: "3.2 Title" / "3.2.1 Title" etc.
        heading_re = re.compile(
            r'^(\d{1,2}(?:\.\d{1,2}){0,3})\s{1,4}([A-Z][A-Za-z0-9 ,()\-/:]{2,80})\s*$',
            re.MULTILINE,
        )
        dot_leader_re = re.compile(r'\.{3,}')

        # Find all heading matches in the full text, skipping TOC page ranges
        hits = []   # (match_start, match_end, sec_num, title)
        for m in heading_re.finditer(full_text):
            if in_skip_range(m.start()):
                continue
            if dot_leader_re.search(m.group(0)):
                continue
            hits.append((m.start(), m.end(), m.group(1), m.group(2).strip()))

        if not hits:
            logger.warning("No section headings found in full text — returning empty list")
            return []

        logger.info(f"Found {len(hits)} section headings in full document text")

        sections: List[ParsedSection] = []
        for i, (start, end, sec_num, title) in enumerate(hits):
            content_start = end
            content_end = hits[i + 1][0] if i + 1 < len(hits) else len(full_text)
            content = full_text[content_start:content_end].strip()

            page_start = char_pos_to_page(start)
            page_end = char_pos_to_page(content_end - 1) if content_end > content_start else page_start

            section = ParsedSection(
                section_number=sec_num,
                title=title,
                content=content,
                page_start=page_start,
                page_end=page_end,
                level=len(sec_num.split('.')),
                parent_section=self._find_parent_section(sec_num, sections),
            )
            sections.append(section)

        return sections

    def _extract_diagrams_by_caption_rendering(
        self,
        doc,
        sections: List[ParsedSection],
        toc_pages: set = None,
    ) -> List[ParsedDiagram]:
        """
        Extract diagrams by finding FIGURE captions on each page and rendering
        the page region above the caption as a PNG pixmap.

        Works for vector/path-based figures that are invisible to get_images().
        Strategy:
          Phase 1 — scan every page (skipping TOC pages) for FIGURE captions that
                    have actual vector drawings above them (excludes header region).
                    For each unique figure number keep the candidate with the MOST
                    drawings (= most likely the actual diagram page, not a reference).
          Phase 2 — render the best candidates in page order, send to Gemini.
        """
        import re as _re

        CAPTION_RE = _re.compile(
            r'((?:FIGURE|FIG\.?)\s+[\d][\d\.\-]+\b.*)',
            _re.IGNORECASE,
        )
        HEADER_BOTTOM = 155.0   # y-coordinate below which header decorations end

        toc_pages = toc_pages or set()
        analyze = self.parser_config.get('analyze_images_with_gemini', True)

        # ── Phase 1: collect candidates ──────────────────────────────────
        # key: fig_num (str) or (page_num, y0_rounded) for unnumbered captions
        # value: (drawing_count, page_num, fig_rect, caption)
        best_candidate: dict = {}

        for page_num in range(len(doc)):
            page_1idx = page_num + 1
            if page_1idx in toc_pages:
                continue

            page = doc[page_num]
            hits = page.search_for("FIGURE ", flags=fitz.TEXT_DEHYPHENATE)
            hits += page.search_for("Figure ", flags=fitz.TEXT_DEHYPHENATE)
            if not hits:
                continue

            page_drawings = page.get_drawings()  # fetch once per page

            for fig_rect in hits:
                cap_area = fitz.Rect(
                    fig_rect.x0,
                    fig_rect.y0,
                    page.rect.x1 - 20,
                    fig_rect.y1 + 90,
                )
                raw = ' '.join(page.get_textbox(cap_area).split())
                m = CAPTION_RE.match(raw)
                if not m:
                    continue
                caption = m.group(1).strip()
                fig_num = self._extract_figure_number(caption)

                # Count drawings in the content area above the caption
                # (excluding header region so header decoration lines don't fool us)
                render_top = max(0.0, fig_rect.y0 - 480)
                fig_content_top = max(render_top, HEADER_BOTTOM)
                figure_area = fitz.Rect(0, fig_content_top, page.rect.x1, fig_rect.y0)
                n_drawings = sum(1 for d in page_drawings if d['rect'].intersects(figure_area))

                if n_drawings == 0:
                    logger.debug(f"Page {page_1idx}: skipping '{caption[:60]}' — no drawings")
                    continue

                # Deduplicate: keep the candidate with the most drawings per fig_num
                key = fig_num if fig_num else (page_num, round(fig_rect.y0))
                existing = best_candidate.get(key)
                if existing is None or n_drawings > existing[0]:
                    best_candidate[key] = (n_drawings, page_num, fig_rect, caption)

        logger.info(f"Caption scan complete — {len(best_candidate)} unique figure candidates")

        # ── Phase 2: render and analyse best candidates in page order ────
        diagrams: List[ParsedDiagram] = []
        sorted_candidates = sorted(best_candidate.values(), key=lambda x: x[1])  # sort by page_num

        for n_drawings, page_num, fig_rect, caption in sorted_candidates:
            page_1idx = page_num + 1
            page = doc[page_num]
            fig_num = self._extract_figure_number(caption)

            render_top = max(0.0, fig_rect.y0 - 480)
            render_rect = fitz.Rect(15, render_top, page.rect.x1 - 15, fig_rect.y1 + 70)
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat, clip=render_rect)
            image_bytes = pix.tobytes("png")

            if len(image_bytes) < 8000:
                continue

            extracted_text, description = "", ""
            if analyze:
                if self._gemini_rate_limited:
                    self._gemini_skipped_count += 1
                    logger.debug(f"Skipping Gemini for diagram on page {page_1idx} (rate-limited)")
                else:
                    result = self._analyze_image_with_gemini(image_bytes, caption)
                    extracted_text = result.get("extracted_text", "")
                    description = result.get("description", "")

            section_number, section_title = None, None
            for sec in sections:
                if sec.page_start <= page_1idx <= sec.page_end:
                    section_number = sec.section_number
                    section_title = sec.title
                    break

            diagram_id = f"diagram_p{page_1idx}_f{fig_num or len(diagrams)}"
            diagrams.append(ParsedDiagram(
                diagram_id=diagram_id,
                caption=caption,
                page=page_1idx,
                figure_number=fig_num,
                image_path=None,
                section_number=section_number,
                section_title=section_title,
                extracted_text=extracted_text,
                diagram_description=description,
                image_size_bytes=len(image_bytes),
            ))
            logger.info(f"Extracted diagram: {diagram_id} — {caption[:70]}")

        if self._gemini_skipped_count > 0:
            logger.info(f"Gemini skipped for {self._gemini_skipped_count} diagrams (rate-limited)")
        return diagrams

    def _process_image_xrefs(
        self,
        doc,
        image_xrefs: dict,
        sections: List[ParsedSection],
    ) -> List[ParsedDiagram]:
        """
        Process pre-collected image xrefs (one per unique image object).
        Filters by requiring a real figure caption nearby; size/dimension are secondary.
        """
        min_size = self.parser_config.get('min_image_size', 10000)
        min_dim  = self.parser_config.get('min_image_dimension', 100)
        analyze  = self.parser_config.get('analyze_images_with_gemini', True)

        diagrams: List[ParsedDiagram] = []

        for xref, page_1idx in tqdm(
            sorted(image_xrefs.items(), key=lambda x: x[1]),
            desc="Processing diagrams",
        ):
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if len(image_bytes) < min_size:
                    continue
                if base_image.get("width", 0) < min_dim or base_image.get("height", 0) < min_dim:
                    continue

                page = doc[page_1idx - 1]
                caption = self._find_image_caption(page, (xref,) + (None,) * 9)
                if not caption:
                    continue

                extracted_text, description = "", ""
                if analyze:
                    if self._gemini_rate_limited:
                        self._gemini_skipped_count += 1
                    else:
                        result = self._analyze_image_with_gemini(image_bytes, caption)
                        extracted_text = result.get("extracted_text", "")
                        description    = result.get("description", "")

                section_number, section_title = None, None
                for sec in sections:
                    if sec.page_start <= page_1idx <= sec.page_end:
                        section_number = sec.section_number
                        section_title  = sec.title
                        break

                diagrams.append(ParsedDiagram(
                    diagram_id=f"diagram_p{page_1idx}_x{xref}",
                    caption=caption,
                    page=page_1idx,
                    figure_number=self._extract_figure_number(caption),
                    image_path=None,
                    section_number=section_number,
                    section_title=section_title,
                    extracted_text=extracted_text,
                    diagram_description=description,
                    image_size_bytes=len(image_bytes),
                ))

            except Exception as e:
                logger.warning(f"Error processing xref {xref}: {e}")

        if self._gemini_analyzed_count > 0 or self._gemini_skipped_count > 0:
            logger.info(
                f"Gemini image analysis: {self._gemini_analyzed_count} analyzed, "
                f"{self._gemini_skipped_count} skipped (rate-limited)"
            )
        return diagrams
    
    def _merge_multipage_tables(self, tables: List[ParsedTable]) -> List[ParsedTable]:
        """
        Merge consecutive table fragments that belong to the same multi-page table.

        Two fragments are merged when:
        - They have the same number of columns, AND
        - The second fragment starts on the page immediately following the first, AND
        - The second fragment has no caption (continuation) OR its caption matches the first.

        The first fragment's caption, table_id, headers, and table_number are kept.
        """
        if not tables:
            return tables

        merged: List[ParsedTable] = []
        i = 0
        while i < len(tables):
            current = tables[i]
            while i + 1 < len(tables):
                nxt = tables[i + 1]
                cur_last_page = max(current.page_range)
                nxt_first_page = min(nxt.page_range)

                same_cols = len(current.headers) == len(nxt.headers)
                adjacent = nxt_first_page == cur_last_page + 1
                # Consider it a continuation if next has no caption or same caption root
                no_caption = not nxt.caption or nxt.caption.startswith("Table ")
                same_caption = (
                    current.table_number
                    and nxt.table_number
                    and current.table_number == nxt.table_number
                )

                if same_cols and adjacent and (no_caption or same_caption):
                    # Merge: extend rows and page_range
                    # Skip first row of continuation if it duplicates the header
                    cont_rows = nxt.rows
                    if cont_rows and cont_rows[0] == current.headers:
                        cont_rows = cont_rows[1:]
                    current = ParsedTable(
                        table_id=current.table_id,
                        caption=current.caption,
                        page_range=sorted(set(current.page_range + nxt.page_range)),
                        headers=current.headers,
                        rows=current.rows + cont_rows,
                        table_number=current.table_number,
                        section_number=current.section_number,
                        section_title=current.section_title,
                        confidence_scores=current.confidence_scores,
                    )
                    i += 1
                else:
                    break
            merged.append(current)
            i += 1

        if len(merged) < len(tables):
            logger.info(
                f"Merged {len(tables)} table fragments into {len(merged)} tables "
                f"(multi-page merging)"
            )
        return merged

    def _extract_sections_from_blocks(
        self,
        all_blocks: List,
        page_texts: List,
    ) -> List[ParsedSection]:
        """
        Extract sections from the full-document block stream.

        Heading detection uses font-size analysis:
        - Compute the modal (most common) font size = body text size.
        - A block is a heading candidate if its dominant font size exceeds body size
          OR if any span in it is bold.
        - The candidate must also match the section-number pattern (e.g. "3.2 Title").

        Content is collected from the raw block stream between consecutive headings,
        so it naturally spans page boundaries without any splitting artefacts.
        Falls back to regex-on-page-text if no headings are found via font analysis.
        """
        import re
        from collections import Counter

        heading_re = re.compile(
            r'^(\d{1,2}(?:\.\d{1,2}){0,3})\s{1,4}([A-Z][A-Za-z0-9 ,()\-/:]{2,80})\s*$'
        )
        dot_leader_re = re.compile(r'\.{3,}')

        # ── Identify TOC pages ────────────────────────────────────────────
        toc_pages = {pn for pn, txt in page_texts if self._is_toc_page(txt)}
        if toc_pages:
            logger.info(f"Skipping {len(toc_pages)} TOC page(s): {sorted(toc_pages)}")

        # ── Compute body font size (modal across all spans) ───────────────
        font_size_counts: Counter = Counter()
        for page_num, block in all_blocks:
            if page_num in toc_pages:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        font_size_counts[round(span.get("size", 0), 1)] += 1

        if not font_size_counts:
            logger.warning("No font data found — falling back to regex section extraction")
            return self._extract_sections_from_pages(page_texts)

        body_size = font_size_counts.most_common(1)[0][0]
        logger.info(f"Detected body font size: {body_size}pt")

        # ── Walk all blocks to find headings ─────────────────────────────
        heading_hits = []  # (page_1idx, sec_num, title, block_idx)

        for block_idx, (page_num, block) in enumerate(all_blocks):
            if page_num in toc_pages:
                continue

            # Aggregate text and dominant font properties of this block
            block_text_parts = []
            max_size = 0.0
            is_bold = False
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    s = span.get("size", 0.0)
                    flags = span.get("flags", 0)
                    if t.strip():
                        block_text_parts.append(t)
                        if s > max_size:
                            max_size = s
                        if flags & (2 ** 4):   # bold bit
                            is_bold = True

            block_text = "".join(block_text_parts).strip()
            if not block_text or dot_leader_re.search(block_text):
                continue

            is_larger = max_size >= body_size + 0.5
            m = heading_re.match(block_text)
            if m and (is_larger or is_bold):
                heading_hits.append((page_num, m.group(1), m.group(2).strip(), block_idx))

        if not heading_hits:
            logger.warning(
                "No headings found via font analysis — falling back to regex section extraction"
            )
            return self._extract_sections_from_pages(page_texts)

        logger.info(f"Found {len(heading_hits)} section headings via font analysis")

        # ── Build sections: collect body text between consecutive headings ─
        sections: List[ParsedSection] = []

        for i, (page_num, sec_num, title, block_idx) in enumerate(heading_hits):
            next_block_idx = (
                heading_hits[i + 1][3] if i + 1 < len(heading_hits) else len(all_blocks)
            )
            end_page = (
                heading_hits[i + 1][0] if i + 1 < len(heading_hits) else all_blocks[-1][0]
            )

            # Collect spans from all blocks between this heading and the next
            content_parts = []
            current_line_parts = []
            for j in range(block_idx + 1, next_block_idx):
                bpg, bblock = all_blocks[j]
                if bpg in toc_pages:
                    continue
                for line in bblock.get("lines", []):
                    current_line_parts = []
                    for span in line.get("spans", []):
                        t = span.get("text", "")
                        if t.strip():
                            current_line_parts.append(t)
                    if current_line_parts:
                        content_parts.append("".join(current_line_parts))

            content = "\n".join(content_parts).strip()

            section = ParsedSection(
                section_number=sec_num,
                title=title,
                content=content,
                page_start=page_num,
                page_end=end_page,
                level=len(sec_num.split(".")),
                parent_section=self._find_parent_section(sec_num, sections),
            )
            sections.append(section)

        logger.info(f"Extracted {len(sections)} sections via full-document block analysis")
        return sections

    def _is_toc_page(self, text: str) -> bool:
        """Return True if this page looks like a Table of Contents page."""
        import re
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return False
        # TOC lines typically contain dot leaders followed by a page number
        toc_pattern = re.compile(r'\.{2,}\s*\d+\s*$')
        toc_hits = sum(1 for l in lines if toc_pattern.search(l))
        return toc_hits / len(lines) > 0.25

    def _extract_sections_from_pages(self, page_texts: List) -> List[ParsedSection]:
        """
        Extract hierarchical sections from per-page text.
        Skips TOC pages and captures the body text that belongs to each section.
        """
        import re
        # Heading pattern: "1" / "1.2" / "1.2.3" followed by a capitalised title
        # Excludes lines with dot leaders (TOC artefacts)
        heading_re = re.compile(
            r'^(\d{1,2}(?:\.\d{1,2}){0,3})\s{1,4}([A-Z][A-Za-z0-9 ,()\-/:]{2,80})\s*$'
        )
        dot_leader_re = re.compile(r'\.{3,}')

        # ── Pass 1: collect (page, section_num, title, char_offset_in_page) for every heading
        heading_hits = []  # list of [page_1idx, section_num, title, line_start_in_page_text]
        for page_num, text in page_texts:
            if self._is_toc_page(text):
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if dot_leader_re.search(stripped):
                    continue
                m = heading_re.match(stripped)
                if m:
                    heading_hits.append((page_num, m.group(1), m.group(2).strip()))

        if not heading_hits:
            logger.warning("No section headings found via PyMuPDF — returning empty section list")
            return []

        sections: List[ParsedSection] = []
        # Build a quick lookup: page_num -> text
        page_map = {pn: txt for pn, txt in page_texts}

        # ── Pass 2: for each heading, collect all body text until the NEXT heading
        for i, (page_num, sec_num, title) in enumerate(heading_hits):
            next_page = heading_hits[i + 1][0] if i + 1 < len(heading_hits) else max(page_map.keys())
            next_title = heading_hits[i + 1][2] if i + 1 < len(heading_hits) else None

            content_parts = []
            for pg in range(page_num, next_page + 1):
                pg_text = page_map.get(pg, "")
                if not pg_text:
                    continue
                lines = pg_text.splitlines()
                # On the heading's own page: skip lines up to and including the heading line
                if pg == page_num:
                    found = False
                    for li, line in enumerate(lines):
                        if not found and title in line:
                            found = True
                            continue
                        if found:
                            content_parts.append(line)
                # On the last page: stop at the next heading line
                elif pg == next_page and next_title:
                    for line in lines:
                        if next_title in line:
                            break
                        content_parts.append(line)
                else:
                    content_parts.extend(lines)

            body = "\n".join(content_parts).strip()

            section = ParsedSection(
                section_number=sec_num,
                title=title,
                content=body,
                page_start=page_num,
                page_end=next_page,
                level=len(sec_num.split('.')),
                parent_section=self._find_parent_section(sec_num, sections)
            )
            sections.append(section)

        logger.info(f"Extracted {len(sections)} sections via PyMuPDF")
        return sections
    
    def parse(self, pdf_path: str, use_fallback: bool = True) -> ParsedDocument:
        """
        Parse PDF using configured parser with fallback.
        
        Args:
            pdf_path: Path to PDF file
            use_fallback: Whether to use fallback parser on failure
            
        Returns:
            ParsedDocument instance
        """
        try:
            if self.primary_parser == 'docling' and HAS_DOCLING:
                return self.parse_with_docling(pdf_path)
            elif self.primary_parser == 'pymupdf' and HAS_PYMUPDF:
                return self.parse_with_pymupdf(pdf_path)
            else:
                logger.warning(f"Primary parser {self.primary_parser} not available")
                if use_fallback:
                    return self._try_fallback(pdf_path)
                raise ValueError(f"Parser {self.primary_parser} not available")
        except Exception as e:
            logger.error(f"Error parsing with {self.primary_parser}: {e}")
            if use_fallback:
                return self._try_fallback(pdf_path)
            raise
    
    def _try_fallback(self, pdf_path: str) -> ParsedDocument:
        """Try fallback parser."""
        logger.info(f"Trying fallback parser: {self.fallback_parser}")
        
        if self.fallback_parser == 'pymupdf' and HAS_PYMUPDF:
            return self.parse_with_pymupdf(pdf_path)
        elif self.fallback_parser == 'docling' and HAS_DOCLING:
            return self.parse_with_docling(pdf_path)
        else:
            raise ValueError(f"No available parser found")
    
    def save_parsed_document(self, parsed_doc: ParsedDocument, output_path: str):
        """
        Save parsed document to JSON.
        
        Args:
            parsed_doc: Parsed document instance
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_doc.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Parsed document saved to {output_path}")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse NASA handbook PDF')
    parser.add_argument('--url', help='URL to download PDF from')
    parser.add_argument('--pdf', help='Path to local PDF file')
    parser.add_argument('--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = get_config()
    pdf_parser = PDFParser(config)
    
    # Determine PDF path
    if args.url:
        pdf_path = pdf_parser.download_pdf(
            args.url,
            config.get('paths.pdf_file')
        )
    elif args.pdf:
        pdf_path = args.pdf
    else:
        # Use configured URL
        url = config.get('download.nasa_handbook_url')
        pdf_path = pdf_parser.download_pdf(url, config.get('paths.pdf_file'))
    
    # Parse PDF
    parsed_doc = pdf_parser.parse(pdf_path)
    
    # Save output
    output_path = args.output or config.get('paths.parsed_output')
    pdf_parser.save_parsed_document(parsed_doc, output_path)
    
    print(f"\n✓ Parsing complete!")
    print(f"  Sections: {len(parsed_doc.sections)}")
    print(f"  Tables: {len(parsed_doc.tables)}")
    print(f"  Diagrams: {len(parsed_doc.diagrams)}")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    main()
