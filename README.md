# NASA Systems Engineering Handbook QA System

A Retrieval-Augmented Generation (RAG) system for querying the NASA Systems Engineering Handbook. Ask natural language questions and get precise, cited answers from the document — including tables, figures, and hierarchical sections.

---

## Architecture

```
PDF (NASA Handbook)
        │
        ▼
┌─────────────────────┐
│   PDF Parser        │  PyMuPDF — extracts text, tables (44),
│   (pdf_parser.py)   │  and vector diagrams (23) by caption detection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Metadata Extractor  │  Acronyms, section hierarchy, cross-references
│ (metadata_extractor)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Hierarchical Chunker│  Token-aware splitting (chunk_size=2000),
│   (chunker.py)      │  section-aware with parent context preserved
└────────┬────────────┘     554 chunks: 487 text + 44 table + 23 diagram
         │
         ▼
┌─────────────────────┐
│   Vector Store      │  sentence-transformers/all-MiniLM-L6-v2 (384-dim)
│  (vector_store.py)  │  FAISS IndexFlatIP (cosine similarity)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Multi-Hop Retriever│  FAISS search → reranking (table caption boost)
│   (retriever.py)    │  → parent section fetch → top-k results
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    QA System        │  Groq LLM (or OpenAI) generates answer with
│   (qa_system.py)    │  citations + citation verification (✅/⚠️/❌)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit UI      │  Chat interface, source viewer, citation badges
│     (app.py)        │
└─────────────────────┘
```

**Key design choices:**
- No LangChain / LlamaIndex — fully custom ingestion and retrieval pipeline
- Vector figures extracted by caption-anchored page rendering (handles PDF path/vector graphics)
- Tables split with headers repeated in every chunk; absolute row numbers preserved
- Citation verification runs post-generation via Jaccard word-overlap scoring

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/realhanokreddy/Hanok_Hireathon_I2E.git
cd Hanok_Hireathon_I2E
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the example config and add your keys:
```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` (or set environment variables):

| Key | Where to get it |
|-----|----------------|
| `GROQ_API_KEY` | https://console.groq.com — free tier |
| `GEMINI_API_KEY` | https://aistudio.google.com — free tier, used for diagram analysis |

### 5. Run the ingestion pipeline *(first time only)*
```bash
python run.py --pipeline
```
This downloads the NASA handbook PDF, parses it, chunks it, and builds the FAISS vector store. Takes ~5–10 minutes.

---

## Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.
