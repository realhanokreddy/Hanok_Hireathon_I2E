# Windows Installation Guide

## ❌ Common Error: "Microsoft Visual C++ 14.0 or greater is required"

This error occurs when pip tries to build packages from source instead of using pre-built wheels.

## ✅ Solution: Install with Pre-built Wheels

### Method 1: Use the Fixed requirements.txt (Recommended)

```bash
# 1. Upgrade pip first
python -m pip install --upgrade pip

# 2. Install with only binary packages (no source builds)
pip install -r requirements.txt --only-binary :all: --prefer-binary

# If that fails, try without the strict flags:
pip install -r requirements.txt --prefer-binary
```

### Method 2: Step-by-Step Installation

If the above still fails, install in stages:

```bash
# Stage 1: Install numpy first (most important)
pip install numpy==1.26.4

# Stage 2: Install core dependencies
pip install pydantic pyyaml python-dotenv requests tqdm pandas

# Stage 3: Install sentence-transformers (will install scikit-learn)
pip install sentence-transformers --prefer-binary

# Stage 4: Install FAISS
pip install faiss-cpu

# Stage 5: Install LLM libraries
pip install groq openai tiktoken

# Stage 6: Install remaining packages
pip install docling pymupdf pypdf2
pip install langchain langchain-openai langchain-community
pip install pytest pytest-cov
```

### Method 3: Use Conda (Alternative)

If pip continues to fail, use Conda which has better pre-built packages:

```bash
# Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
# Then:

conda create -n nasa_qa python=3.11
conda activate nasa_qa
conda install -c conda-forge numpy pandas scikit-learn faiss-cpu
pip install -r requirements.txt --no-deps  # Install remaining packages
```

## 🔧 Troubleshooting

### Error: "No matching distribution found"

```bash
# Update pip and setuptools
python -m pip install --upgrade pip setuptools wheel
```

### Error: Still getting C++ compiler errors

**Option A: Install pre-built wheel directly**
```bash
# For scikit-learn
pip install scikit-learn --only-binary :all:

# For sentence-transformers
pip install sentence-transformers --only-binary scikit-learn
```

**Option B: Install Microsoft C++ Build Tools** (large download, ~6GB)
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer
3. Select "C++ build tools"
4. Install
5. Restart terminal
6. Try `pip install -r requirements.txt` again

### Error: "Cannot uninstall package"

```bash
# Force reinstall
pip install --force-reinstall --no-cache-dir <package-name>
```

## ✅ Verify Installation

After installation completes:

```bash
# Test imports
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import sentence_transformers; print('sentence_transformers: OK')"
python -c "import faiss; print('faiss: OK')"
python -c "import groq; print('groq: OK')"
python -c "from docling.document_converter import DocumentConverter; print('docling: OK')"

# Or run the test script
python -c "import sys; sys.path.append('.'); from env.config_loader import get_env_config; print('All imports successful!')"
```

## 📦 Minimal Installation (If Still Having Issues)

Install only the essentials needed to run:

```bash
# Absolute minimum for free tier
pip install numpy pandas pydantic pyyaml python-dotenv requests tqdm
pip install sentence-transformers faiss-cpu
pip install groq tiktoken
pip install pymupdf  # Simpler than docling

# Skip these if they cause issues:
# - docling (can use pymupdf only)
# - langchain (not strictly required)
# - openai (only if using groq)
```

Then update `env/.env`:
```bash
PDF_PRIMARY_PARSER=pymupdf  # Instead of docling
```

## 🎯 Quick Fix Command (Try This First!)

```bash
# Clear pip cache and reinstall with pre-built wheels only
pip cache purge
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --prefer-binary --no-cache-dir
```

## 💡 Why This Happens

Some Python packages have C extensions that need to be compiled:
- `scikit-learn` (dependency of sentence-transformers)
- `numpy` 
- `scipy`
- `faiss-cpu`

On Windows, pip tries to build these from source if pre-built wheels aren't available. The solution is to:
1. Use newer package versions that have pre-built wheels
2. Install from official wheel repositories
3. Use `--prefer-binary` flag
4. Or use Conda which handles this better

## 🚀 Recommended: Fresh Installation

If you've had multiple failed attempts:

```bash
# 1. Delete virtual environment
rmdir /s venv

# 2. Create fresh venv
python -m venv venv
venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install with binary preference
pip install -r requirements.txt --prefer-binary

# 5. If still fails, use step-by-step method above
```

## 📝 Package Notes

### sentence-transformers
- Depends on: scikit-learn, scipy, transformers, torch
- Pre-built wheels available for Python 3.8-3.11
- If installation fails, try: `pip install sentence-transformers==2.5.1 --prefer-binary`

### docling
- Newer package, may not have wheels for all Python versions
- Fallback: Use pymupdf instead (simpler, always has wheels)

### faiss-cpu
- Pre-built wheels available for Windows
- Make sure to use `faiss-cpu` not `faiss` (faiss requires compilation)

## 🆘 Still Having Issues?

1. Check Python version: `python --version` (should be 3.10 or 3.11)
2. Check pip version: `pip --version` (should be 23.0+)
3. Try the minimal installation above
4. Use Conda instead of pip
5. Or use WSL2 (Windows Subsystem for Linux) which handles this better

## ✅ Success Checklist

- [ ] Python 3.10 or 3.11 installed
- [ ] pip version 23.0+
- [ ] Virtual environment activated
- [ ] pip cache cleared
- [ ] Using `--prefer-binary` flag
- [ ] All imports work (run verify command above)
