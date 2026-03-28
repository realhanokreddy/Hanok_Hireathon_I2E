# NASA Technical Manual QA System

A production-ready RAG system for complex technical documents with advanced PDF parsing, hierarchical chunking, and multi-hop question answering. **Free tier compatible** using Groq and local embeddings!

## Features

- **100% Free Tier Support**: Uses Groq's free API (60 req/min) + local sentence-transformers embeddings
- **Advanced PDF Parsing**: Uses Docling for superior table extraction (95% accuracy), layout analysis, and structure preservation
- **Hierarchical Chunking**: Respects section boundaries and maintains parent-child relationships
- **Multi-hop Question Answering**: Automatically resolves cross-references across chapters
- **Precise Citations**: Returns section numbers, page numbers, and confidence scores
- **Table Extraction**: Handles multi-page tables with header preservation
- **Acronym Resolution**: Automatically expands acronyms for better retrieval
- **Flexible Providers**: Choose between Groq (free) or OpenAI (paid) for LLM, local or OpenAI for embeddings

## Target Document

NASA Systems Engineering Handbook (SP-2016-6105 Rev2)
- Source: https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf
- Pages: ~270
- Content: Process diagrams, decision trees, multi-page tables, heavy cross-referencing

## Project Structure

```
.
├── data/                       # NASA PDF and parsed outputs
├── src/
│   ├── pdf_parser.py          # Docling-based PDF parsing
│   ├── metadata_extractor.py  # Extract acronyms, hierarchy, cross-refs
│   ├── chunker.py             # Hierarchical chunking logic
│   ├── vector_store.py        # Embedding and vector DB
│   ├── retriever.py           # Multi-hop retrieval
│   └── qa_system.py           # Main QA interface
├── tests/                     # Test queries and verification
├── config.yaml                # Configuration
├── requirements.txt           # Dependencies
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Quick Start (Free Tier)

**Option 1: Interactive Setup (Recommended)**
```bash
python setup_env.py
```
This will guide you through setting up your configuration interactively.

**Option 2: Manual Setup**
1. Get a free Groq API key from https://console.groq.com
2. Copy `env/.env.example` to `env/.env` and add your API key:
   ```bash
   copy env\.env.example env\.env
   # Edit env/.env and set: GROQ_API_KEY=gsk_your_key_here
   ```
3. Use default config (already set for free tier)

**Option 3: Use Root .env (Legacy)**
1. Copy `.env.example` to `.env` and add:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Configuration Documentation
- **Quick Reference**: [ENV_QUICK_REFERENCE.md](ENV_QUICK_REFERENCE.md) - Common scenarios and cheat sheet
- **Full Guide**: [env/README.md](env/README.md) - Complete documentation
- **All Variables**: [env/.env.example](env/.env.example) - Template with all options

### Alternative: OpenAI Setup

1. Edit `env/.env` and set:
   ```bash
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk_your_key_here
   EMBEDDING_PROVIDER=openai
   VECTOR_STORE_DIMENSION=3072
   ```
2. Rebuild vector store: `python run.py --pipeline --skip-download --skip-parse`

## Usage

### Full Pipeline (Recommended for First Run)

```bash
# Download PDF, parse, chunk, and build vector store
python run.py --pipeline

# Query the system
python run.py --query "What are the entry criteria for PDR?"

# Interactive mode
python run.py --interactive
```

### Individual Steps

```bash
# Just download and parse
python run.py --pipeline --skip-chunk --skip-vector

# Just build vector store (after parsing)
python run.py --pipeline --skip-download --skip-parse

# Test with sample queries
python run.py --test

# Evaluate all test queries
python run.py --evaluate
```

## Example Queries

- "What does the systems engineering process flow look like?"
- "What are the entry criteria for PDR?"
- "How does the risk management process feed into the technical review process?"
- "What is a TRL and what are its levels?"

## Requirements

Minimum:
- Python 3.10+
- 8GB RAM
- Internet connection for API calls

Recommended:
- Python 3.11+
- 16GB RAM
- GPU for faster processing (optional)

## License

MIT License

## Acknowledgments

- Docling by IBM Research for PDF parsing
- NASA for the Systems Engineering Handbook (public domain)
