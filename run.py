"""
Main run script for the NASA Technical Manual QA System.
Provides a simple interface to run the full pipeline.
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.vector_store import VectorStore
from src.generation.qa_system import TechnicalQASystem


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/qa_system.log'),
            logging.StreamHandler()
        ]
    )


def run_pipeline(skip_download: bool = False, skip_parse: bool = False,
                skip_chunk: bool = False, skip_vector: bool = False):
    """
    Run the full document processing pipeline.
    
    Args:
        skip_download: Skip PDF download if already exists
        skip_parse: Skip parsing if already done
        skip_chunk: Skip chunking if already done
        skip_vector: Skip vector store build if already done
    """
    config = get_config()
    
    print("\n" + "=" * 80)
    print("RUNNING FULL DOCUMENT PROCESSING PIPELINE")
    print("=" * 80 + "\n")
    
    # Step 1: Download and Parse PDF
    if not skip_download or not skip_parse:
        print("STEP 1: Downloading and Parsing Clinical Research Operations Manual PDF")
        print("-" * 80)
        
        parser = PDFParser(config)
        
        # Check if PDF exists
        pdf_path = Path(config.get('paths.pdf_file'))
        if skip_download and pdf_path.exists():
            print(f"[OK] Using existing PDF: {pdf_path}")
        else:
            # Download PDF
            url = config.get('download.clinical_research_manual_url')
            print(f"Downloading from: {url}")
            pdf_path = parser.download_pdf(url, str(pdf_path))
            print(f"[OK] Downloaded to: {pdf_path}")
        
        # Parse PDF
        if not skip_parse:
            print("\nParsing PDF with Docling...")
            parsed_doc = parser.parse(str(pdf_path))
            
            # Save parsed output
            output_path = config.get('paths.parsed_output')
            parser.save_parsed_document(parsed_doc, output_path)
            
            print(f"[OK] Parsed document:")
            print(f"  - Sections: {len(parsed_doc.sections)}")
            print(f"  - Tables: {len(parsed_doc.tables)}")
            print(f"  - Diagrams: {len(parsed_doc.diagrams)}")
            print(f"  - Output: {output_path}")
        else:
            print("[OK] Skipping parse (using existing parsed output)")
    else:
        print("STEP 1: Skipped (using existing PDF and parsed output)")
    
    # Step 2: Extract Metadata
    print("\n\nSTEP 2: Extracting Metadata")
    print("-" * 80)
    
    import json
    from src.ingestion.pdf_parser import ParsedDocument, ParsedSection, ParsedTable, ParsedDiagram
    
    # Load parsed document
    parsed_output = config.get('paths.parsed_output')
    with open(parsed_output, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
    
    sections = [ParsedSection(**s) for s in parsed_data.get('sections', [])]
    tables = [ParsedTable(**t) for t in parsed_data.get('tables', [])]
    diagrams = [ParsedDiagram(**d) for d in parsed_data.get('diagrams', [])]
    
    parsed_doc = ParsedDocument(
        title=parsed_data['title'],
        sections=sections,
        tables=tables,
        diagrams=diagrams,
        metadata=parsed_data['metadata'],
        raw_text=parsed_data.get('raw_text', '')
    )
    
    extractor = MetadataExtractor(config)
    metadata = extractor.extract_all_metadata(parsed_doc)
    
    print(f"[OK] Metadata extracted:")
    print(f"  - Acronyms: {len(metadata.get('acronyms', {}))}")
    print(f"  - Section hierarchy: {len(metadata.get('section_hierarchy', {}))}")
    print(f"  - Cross-references: {len(metadata.get('cross_references', []))}")
    
    # Step 3: Chunk Document
    if not skip_chunk:
        print("\n\nSTEP 3: Hierarchical Chunking")
        print("-" * 80)
        
        chunker = HierarchicalChunker(config)
        chunks = chunker.chunk_document(parsed_doc, metadata, metadata.get('acronyms'))
        
        # Save chunks
        chunks_output = config.get('paths.chunks_output')
        chunker.save_chunks(chunks, chunks_output)
        
        print(f"[OK] Document chunked:")
        print(f"  - Total chunks: {len(chunks)}")
        print(f"  - Text chunks: {sum(1 for c in chunks if c.chunk_type == 'text')}")
        print(f"  - Table chunks: {sum(1 for c in chunks if c.chunk_type == 'table')}")
        print(f"  - Diagram chunks: {sum(1 for c in chunks if c.chunk_type == 'diagram_ref')}")
        print(f"  - Output: {chunks_output}")
    else:
        print("\n\nSTEP 3: Skipped (using existing chunks)")
    
    # Step 4: Build Vector Store
    if not skip_vector:
        print("\n\nSTEP 4: Building Vector Store")
        print("-" * 80)
        
        # Load chunks
        chunks_path = config.get('paths.chunks_output')
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        from src.ingestion.chunker import DocumentChunk
        chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
        
        # Build index
        vector_store = VectorStore(config)
        print(f"Embedding {len(chunks)} chunks (this may take a few minutes)...")
        vector_store.build_index(chunks)
        
        # Save
        save_path = config.get('paths.vector_store')
        vector_store.save(save_path)
        
        print(f"[OK] Vector store built:")
        print(f"  - Chunks indexed: {len(chunks)}")
        print(f"  - Saved to: {save_path}")
    else:
        print("\n\nSTEP 4: Skipped (using existing vector store)")
    
    print("\n\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nYou can now run the QA system:")
    print("  python run.py --query 'Your question here'")
    print("  python run.py --interactive")
    print("  python run.py --test")
    print()


def run_qa_system(query: str = None, interactive: bool = False, test: bool = False):
    """
    Run the QA system.
    
    Args:
        query: Single query to answer
        interactive: Run in interactive mode
        test: Run test queries
    """
    config = get_config()
    
    print("\nInitializing QA system...")
    qa_system = TechnicalQASystem(config=config)
    print("[OK] QA system ready!\n")
    
    if query:
        # Single query
        print(f"Q: {query}\n")
        result = qa_system.ask(query)
        print(qa_system.format_answer(result))
    
    elif test:
        # Test queries
        test_queries = [
            "What are the entry criteria for PDR?",
            "How does risk management feed into the technical review process?",
            "What is TRL and what are its levels?",
            "What does the systems engineering process flow look like?",
        ]
        
        for i, q in enumerate(test_queries, 1):
            print(f"\n{'=' * 80}")
            print(f"Test Query {i}/{len(test_queries)}")
            print(f"{'=' * 80}\n")
            
            result = qa_system.ask(q)
            print(qa_system.format_answer(result))
    
    elif interactive:
        # Interactive mode
        print("Interactive QA mode (type 'quit' to exit)\n")
        
        while True:
            try:
                q = input("\nQ: ").strip()
                
                if q.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not q:
                    continue
                
                result = qa_system.ask(q)
                print(f"\n{qa_system.format_answer(result)}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        print("\nGoodbye!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='clinical research operations manual question-answering system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run.py --pipeline
  
  # Ask a question
  python run.py --query "What are the entry criteria for PDR?"
  
  # Interactive mode
  python run.py --interactive
  
  # Run test queries
  python run.py --test
  
  # Run evaluation
  python run.py --evaluate
        """
    )
    
    parser.add_argument('--pipeline', action='store_true',
                       help='Run full document processing pipeline')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip PDF download (use existing)')
    parser.add_argument('--skip-parse', action='store_true',
                       help='Skip PDF parsing (use existing)')
    parser.add_argument('--skip-chunk', action='store_true',
                       help='Skip chunking (use existing)')
    parser.add_argument('--skip-vector', action='store_true',
                       help='Skip vector store build (use existing)')
    
    parser.add_argument('--query', type=str,
                       help='Single query to answer')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive QA mode')
    parser.add_argument('--test', action='store_true',
                       help='Run test queries')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation on test queries')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        if args.pipeline:
            # Run pipeline
            run_pipeline(
                skip_download=args.skip_download,
                skip_parse=args.skip_parse,
                skip_chunk=args.skip_chunk,
                skip_vector=args.skip_vector
            )
        
        elif args.evaluate:
            # Run evaluation
            from tests.test_evaluation import main as eval_main
            eval_main()
        
        elif args.query or args.interactive or args.test:
            # Run QA system
            run_qa_system(
                query=args.query,
                interactive=args.interactive,
                test=args.test
            )
        
        else:
            # Show help
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
