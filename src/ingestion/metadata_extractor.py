"""
Metadata extraction for NASA technical documents.
Extracts acronyms, cross-references, section hierarchy, and document structure.
"""
import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from src.config import get_config
from src.ingestion.pdf_parser import ParsedDocument, ParsedSection


logger = logging.getLogger(__name__)


@dataclass
class AcronymDefinition:
    """Represents an acronym and its definition."""
    acronym: str
    definition: str
    first_occurrence_page: int
    frequency: int = 1


@dataclass
class CrossReference:
    """Represents a cross-reference between sections."""
    source_section: str
    target_section: str
    reference_type: str  # 'explicit', 'implicit', 'see_also'
    context: str


@dataclass
class SectionHierarchy:
    """Represents the hierarchical structure of sections."""
    section_number: str
    title: str
    level: int
    parent: Optional[str]
    children: List[str]
    path: List[str]  # Full path from root


class MetadataExtractor:
    """Extract metadata from parsed documents."""
    
    def __init__(self, config=None):
        """
        Initialize metadata extractor.
        
        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self.metadata_config = self.config.get_metadata_config()
        
        # Comprehensive NASA/aerospace acronym dictionary
        self.known_acronyms = {
            # Key Decision Points & Reviews
            'TRL': 'Technology Readiness Level',
            'KDP': 'Key Decision Point',
            'SRR': 'System Requirements Review',
            'PDR': 'Preliminary Design Review',
            'CDR': 'Critical Design Review',
            'SIR': 'System Integration Review',
            'ORR': 'Operational Readiness Review',
            'FRR': 'Flight Readiness Review',
            'SAR': 'System Acceptance Review',
            'TRR': 'Test Readiness Review',
            'PRR': 'Production Readiness Review',
            'MRR': 'Mission Readiness Review',
            'MDR': 'Mission Definition Review',
            'SRR': 'Software Requirements Review',
            'SDR': 'System Design Review',
            'DAR': 'Design Analysis Review',
            
            # Systems Engineering & Management
            'SE': 'Systems Engineering',
            'SEMP': 'Systems Engineering Management Plan',
            'RMP': 'Risk Management Plan',
            'IMP': 'Integrated Master Plan',
            'IMS': 'Integrated Master Schedule',
            'WBS': 'Work Breakdown Structure',
            'MBSE': 'Model-Based Systems Engineering',
            'TPM': 'Technical Performance Measure',
            'MOE': 'Measure of Effectiveness',
            'MOP': 'Measure of Performance',
            'KPP': 'Key Performance Parameter',
            'CONOPS': 'Concept of Operations',
            'OCD': 'Operational Concept Description',
            
            # Requirements & Specifications
            'ORD': 'Operational Requirements Document',
            'SRD': 'System Requirements Document',
            'SSRD': 'System/Subsystem Requirements Document',
            'ICD': 'Interface Control Document',
            'IDD': 'Interface Design Document',
            'DID': 'Data Item Description',
            'CDD': 'Capability Development Document',
            'CPD': 'Capability Production Document',
            
            # Configuration & Data Management
            'CM': 'Configuration Management',
            'CCB': 'Configuration Control Board',
            'CMMI': 'Capability Maturity Model Integration',
            'PDR': 'Product Data Repository',
            'PLM': 'Product Lifecycle Management',
            'VCS': 'Version Control System',
            
            # Testing & Verification
            'IV&V': 'Independent Verification and Validation',
            'V&V': 'Verification and Validation',
            'ATP': 'Acceptance Test Procedure',
            'QA': 'Quality Assurance',
            'QC': 'Quality Control',
            'SQA': 'Software Quality Assurance',
            'DT': 'Development Testing',
            'OT': 'Operational Testing',
            'FMEA': 'Failure Mode and Effects Analysis',
            'FMECA': 'Failure Modes, Effects, and Criticality Analysis',
            'FTA': 'Fault Tree Analysis',
            
            # Risk & Safety
            'PRA': 'Probabilistic Risk Assessment',
            'SIL': 'Safety Integrity Level',
            'HAZOP': 'Hazard and Operability Study',
            'MORT': 'Management Oversight and Risk Tree',
            'CRM': 'Continuous Risk Management',
            
            # Project Management
            'PM': 'Project Manager',
            'PMO': 'Program Management Office',
            'SOW': 'Statement of Work',
            'PWS': 'Performance Work Statement',
            'CLIN': 'Contract Line Item Number',
            'EVM': 'Earned Value Management',
            'BCWS': 'Budgeted Cost of Work Scheduled',
            'BCWP': 'Budgeted Cost of Work Performed',
            'ACWP': 'Actual Cost of Work Performed',
            'EAC': 'Estimate at Completion',
            'ETC': 'Estimate to Complete',
            'CPI': 'Cost Performance Index',
            'SPI': 'Schedule Performance Index',
            
            # Organizations & Standards
            'NASA': 'National Aeronautics and Space Administration',
            'DOD': 'Department of Defense',
            'ESA': 'European Space Agency',
            'IEEE': 'Institute of Electrical and Electronics Engineers',
            'ISO': 'International Organization for Standardization',
            'INCOSE': 'International Council on Systems Engineering',
            'AIAA': 'American Institute of Aeronautics and Astronautics',
            'SAE': 'Society of Automotive Engineers',
            'AS': 'Aerospace Standard',
            'MIL-STD': 'Military Standard',
            
            # Technical Terms
            'CAD': 'Computer-Aided Design',
            'CAE': 'Computer-Aided Engineering',
            'CAM': 'Computer-Aided Manufacturing',
            'CFD': 'Computational Fluid Dynamics',
            'FEA': 'Finite Element Analysis',
            'COTS': 'Commercial Off-The-Shelf',
            'GOTS': 'Government Off-The-Shelf',
            'MOTS': 'Modified Off-The-Shelf',
            'NDI': 'Non-Developmental Item',
            'API': 'Application Programming Interface',
            'GUI': 'Graphical User Interface',
            'CLI': 'Command Line Interface',
            
            # Lifecycle & Development
            'SLC': 'System Life Cycle',
            'SDLC': 'Software Development Life Cycle',
            'PDR': 'Preliminary Design Review',
            'LORA': 'Level of Repair Analysis',
            'RAM': 'Reliability, Availability, and Maintainability',
            'RMA': 'Reliability, Maintainability, and Availability',
            'MTBF': 'Mean Time Between Failures',
            'MTTR': 'Mean Time To Repair',
            'MTTF': 'Mean Time To Failure',
            
            # Mission & Operations
            'MOC': 'Mission Operations Center',
            'SOC': 'Science Operations Center',
            'FOD': 'Flight Operations Directorate',
            'ISS': 'International Space Station',
            'LEO': 'Low Earth Orbit',
            'GEO': 'Geostationary Earth Orbit',
            'MEO': 'Medium Earth Orbit',
            'EVA': 'Extravehicular Activity',
            'RCS': 'Reaction Control System',
            'GNC': 'Guidance, Navigation, and Control',
            'ADCS': 'Attitude Determination and Control System',
            'TCS': 'Thermal Control System',
            'EPS': 'Electrical Power System',
            'C&DH': 'Command and Data Handling',
            'TT&C': 'Telemetry, Tracking, and Command',
        }
    
    def extract_all_metadata(self, parsed_doc: ParsedDocument) -> Dict:
        """
        Extract all metadata from parsed document.
        
        Args:
            parsed_doc: Parsed document instance
            
        Returns:
            Dictionary with all extracted metadata
        """
        logger.info("Extracting metadata from document")
        
        metadata = {
            'document_title': parsed_doc.title,
            'total_sections': len(parsed_doc.sections),
            'total_tables': len(parsed_doc.tables),
            'total_diagrams': len(parsed_doc.diagrams),
        }
        
        if self.metadata_config.get('extract_acronyms', True):
            metadata['acronyms'] = self.extract_acronyms(parsed_doc)
        
        if self.metadata_config.get('extract_section_hierarchy', True):
            metadata['section_hierarchy'] = self.build_section_hierarchy(parsed_doc.sections)
        
        if self.metadata_config.get('resolve_cross_references', True):
            metadata['cross_references'] = self.extract_cross_references(parsed_doc)
        
        logger.info(f"Metadata extraction complete: {len(metadata.get('acronyms', {}))} acronyms, "
                   f"{len(metadata.get('cross_references', []))} cross-references")
        
        return metadata
    
    def extract_acronyms(self, parsed_doc: ParsedDocument) -> Dict[str, AcronymDefinition]:
        """
        Extract acronyms and their definitions from the document.
        
        Args:
            parsed_doc: Parsed document instance
            
        Returns:
            Dictionary mapping acronyms to their definitions
        """
        logger.info("Extracting acronyms")
        
        acronyms = {}
        
        # Start with known acronyms
        for acronym, definition in self.known_acronyms.items():
            acronyms[acronym] = AcronymDefinition(
                acronym=acronym,
                definition=definition,
                first_occurrence_page=0,
                frequency=0
            )
        
        # Enhanced patterns for acronym detection
        # Match: "Technology Readiness Level (TRL)"
        definition_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(([A-Z]{2,})\)'
        # Also match: "(TRL) Technology Readiness Level" (reverse format)
        reverse_pattern = r'\(([A-Z]{2,})\)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        # Match standalone acronyms (2+ uppercase letters)
        acronym_pattern = r'\b([A-Z]{2,})\b'
        
        # First pass: Find definitions in sections (to get page numbers)
        for section in parsed_doc.sections:
            section_text = section.content
            
            # Find forward definitions: "Technology Readiness Level (TRL)"
            for match in re.finditer(definition_pattern, section_text):
                definition = match.group(1)
                acronym = match.group(2)
                
                if acronym not in acronyms:
                    acronyms[acronym] = AcronymDefinition(
                        acronym=acronym,
                        definition=definition,
                        first_occurrence_page=section.page_start,
                        frequency=0
                    )
                elif acronyms[acronym].first_occurrence_page == 0:
                    # Update page for known acronyms when first found
                    acronyms[acronym].first_occurrence_page = section.page_start
                    acronyms[acronym].definition = definition  # Update with document's definition
            
            # Find reverse definitions: "(TRL) Technology Readiness Level"
            for match in re.finditer(reverse_pattern, section_text):
                acronym = match.group(1)
                definition = match.group(2)
                
                if acronym not in acronyms:
                    acronyms[acronym] = AcronymDefinition(
                        acronym=acronym,
                        definition=definition,
                        first_occurrence_page=section.page_start,
                        frequency=0
                    )
                elif acronyms[acronym].first_occurrence_page == 0:
                    acronyms[acronym].first_occurrence_page = section.page_start
                    acronyms[acronym].definition = definition
        
        # Second pass: Count frequencies in all text
        text = parsed_doc.raw_text
        for match in re.finditer(acronym_pattern, text):
            acronym = match.group(1)
            if acronym in acronyms:
                acronyms[acronym].frequency += 1
        
        # Filter out very common words that might be false positives
        common_words = {'AND', 'THE', 'FOR', 'NOT', 'BUT', 'ARE', 'WAS', 'HAS', 'HAD', 'CAN', 'MAY', 
                       'USE', 'ALL', 'ANY', 'NEW', 'ONE', 'TWO', 'GET', 'SET', 'RUN', 'END', 'OUT'}
        acronyms = {k: v for k, v in acronyms.items() if k not in common_words}
        
        logger.info(f"Found {len(acronyms)} acronyms")
        return acronyms
    
    def build_section_hierarchy(self, sections: List[ParsedSection]) -> Dict[str, SectionHierarchy]:
        """
        Build hierarchical structure of sections.
        
        Args:
            sections: List of parsed sections
            
        Returns:
            Dictionary mapping section numbers to hierarchy info
        """
        logger.info("Building section hierarchy")
        
        hierarchy = {}
        
        for section in sections:
            # Find children
            children = [
                s.section_number for s in sections
                if s.parent_section == section.section_number
            ]
            
            # Build path from root to this section
            path = self._build_section_path(section.section_number, sections)
            
            hierarchy[section.section_number] = SectionHierarchy(
                section_number=section.section_number,
                title=section.title,
                level=section.level,
                parent=section.parent_section,
                children=children,
                path=path
            )
        
        logger.info(f"Built hierarchy for {len(hierarchy)} sections")
        return hierarchy
    
    def _build_section_path(self, section_number: str, sections: List[ParsedSection]) -> List[str]:
        """Build path from root to given section."""
        path = []
        current = section_number
        
        # Walk up the parent chain
        section_dict = {s.section_number: s for s in sections}
        
        while current:
            path.insert(0, current)
            section = section_dict.get(current)
            if section and section.parent_section:
                current = section.parent_section
            else:
                break
        
        return path
    
    def extract_cross_references(self, parsed_doc: ParsedDocument) -> List[CrossReference]:
        """
        Extract cross-references between sections.
        
        Args:
            parsed_doc: Parsed document instance
            
        Returns:
            List of cross-references
        """
        logger.info("Extracting cross-references")
        
        cross_refs = []
        
        # Patterns for cross-references
        patterns = [
            # "see Section 6.3"
            (r'see\s+Section\s+(\d+(?:\.\d+)*)', 'explicit'),
            # "as described in Section 6.3"
            (r'(?:described|discussed|shown|defined)\s+in\s+Section\s+(\d+(?:\.\d+)*)', 'explicit'),
            # "Section 6.3.2"
            (r'Section\s+(\d+(?:\.\d+)*)', 'implicit'),
            # "Chapter 6"
            (r'Chapter\s+(\d+)', 'implicit'),
            # "refer to Section 6.3"
            (r'refer\s+to\s+Section\s+(\d+(?:\.\d+)*)', 'see_also'),
        ]
        
        # Search in each section
        for section in parsed_doc.sections:
            source_section = section.section_number
            content = section.content + " " + section.title
            
            for pattern, ref_type in patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    target_section = match.group(1)
                    
                    # Get context (words around the match)
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end]
                    
                    cross_ref = CrossReference(
                        source_section=source_section,
                        target_section=target_section,
                        reference_type=ref_type,
                        context=context.strip()
                    )
                    cross_refs.append(cross_ref)
        
        # Remove duplicates
        cross_refs = self._deduplicate_cross_refs(cross_refs)
        
        logger.info(f"Found {len(cross_refs)} cross-references")
        return cross_refs
    
    def _deduplicate_cross_refs(self, cross_refs: List[CrossReference]) -> List[CrossReference]:
        """Remove duplicate cross-references."""
        seen = set()
        unique_refs = []
        
        for ref in cross_refs:
            key = (ref.source_section, ref.target_section, ref.reference_type)
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        
        return unique_refs
    
    def get_section_context(
        self,
        section_number: str,
        sections: List[ParsedSection],
        hierarchy: Dict[str, SectionHierarchy]
    ) -> Dict[str, any]:
        """
        Get contextual information for a section.
        
        Args:
            section_number: Section number to get context for
            sections: List of all sections
            hierarchy: Section hierarchy mapping
            
        Returns:
            Dictionary with contextual information
        """
        if section_number not in hierarchy:
            return {}
        
        section_info = hierarchy[section_number]
        section_dict = {s.section_number: s for s in sections}
        
        # Get parent context
        parent_titles = []
        for section_num in section_info.path[:-1]:  # Exclude current section
            if section_num in section_dict:
                parent_titles.append(section_dict[section_num].title)
        
        # Get sibling sections
        siblings = []
        if section_info.parent:
            parent = hierarchy.get(section_info.parent)
            if parent:
                siblings = [c for c in parent.children if c != section_number]
        
        return {
            'section_number': section_number,
            'title': section_info.title,
            'level': section_info.level,
            'parent': section_info.parent,
            'parent_titles': parent_titles,
            'path': section_info.path,
            'children': section_info.children,
            'siblings': siblings,
            'breadcrumb': ' > '.join([
                section_dict[num].title for num in section_info.path if num in section_dict
            ])
        }
    
    def expand_acronyms_in_text(self, text: str, acronyms: Dict[str, AcronymDefinition]) -> str:
        """
        Expand acronyms in text for better embedding.
        
        Args:
            text: Input text
            acronyms: Dictionary of acronyms
            
        Returns:
            Text with expanded acronyms
        """
        expanded_text = text
        
        for acronym, definition in acronyms.items():
            # Replace first occurrence with expansion
            pattern = r'\b' + re.escape(acronym) + r'\b'
            replacement = f"{definition.definition} ({acronym})"
            
            # Only replace first occurrence in each text
            expanded_text = re.sub(pattern, replacement, expanded_text, count=1)
        
        return expanded_text
    
    def get_cross_references_for_section(
        self,
        section_number: str,
        cross_refs: List[CrossReference]
    ) -> Dict[str, List[str]]:
        """
        Get cross-references related to a specific section.
        
        Args:
            section_number: Section number
            cross_refs: List of all cross-references
            
        Returns:
            Dictionary with outgoing and incoming references
        """
        outgoing = []  # References from this section to others
        incoming = []  # References from other sections to this one
        
        for ref in cross_refs:
            if ref.source_section == section_number:
                outgoing.append(ref.target_section)
            if ref.target_section == section_number:
                incoming.append(ref.source_section)
        
        return {
            'outgoing': list(set(outgoing)),
            'incoming': list(set(incoming)),
            'total_out': len(outgoing),
            'total_in': len(incoming)
        }


def main():
    """Main function for testing."""
    import json
    from src.pdf_parser import PDFParser
    
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
    
    # Reconstruct ParsedDocument (simplified for testing)
    from src.pdf_parser import ParsedDocument, ParsedSection
    
    sections = [
        ParsedSection(**s) for s in parsed_data.get('sections', [])
    ]
    
    parsed_doc = ParsedDocument(
        title=parsed_data['title'],
        sections=sections,
        tables=[],
        diagrams=[],
        metadata=parsed_data['metadata'],
        raw_text=parsed_data.get('raw_text', '')
    )
    
    # Extract metadata
    extractor = MetadataExtractor(config)
    metadata = extractor.extract_all_metadata(parsed_doc)
    
    # Save metadata
    metadata_output = config.get('paths.data_dir') + '/metadata.json'
    
    # Convert to serializable format
    serializable_metadata = {
        'document_title': metadata['document_title'],
        'total_sections': metadata['total_sections'],
        'total_tables': metadata['total_tables'],
        'total_diagrams': metadata['total_diagrams'],
        'acronyms': {k: v.__dict__ for k, v in metadata.get('acronyms', {}).items()},
        'section_hierarchy': {k: v.__dict__ for k, v in metadata.get('section_hierarchy', {}).items()},
        'cross_references': [ref.__dict__ for ref in metadata.get('cross_references', [])]
    }
    
    with open(metadata_output, 'w') as f:
        json.dump(serializable_metadata, f, indent=2)
    
    print(f"\n✓ Metadata extraction complete!")
    print(f"  Acronyms: {len(metadata.get('acronyms', {}))}")
    print(f"  Sections: {len(metadata.get('section_hierarchy', {}))}")
    print(f"  Cross-references: {len(metadata.get('cross_references', []))}")
    print(f"  Output: {metadata_output}")


if __name__ == '__main__':
    main()
