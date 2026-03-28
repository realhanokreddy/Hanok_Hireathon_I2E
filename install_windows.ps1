# Windows Installation Script for NASA QA System
# Automatically handles package installation with proper error handling

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   NASA QA System - Windows Installation" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

if ($pythonVersion -match "Python 3\.(10|11|12)") {
    Write-Host "✓ Python version is compatible" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: Python 3.10-3.12 recommended. You have: $pythonVersion" -ForegroundColor Yellow
}

Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel --quiet
Write-Host "✓ pip upgraded" -ForegroundColor Green

Write-Host ""

# Clear pip cache
Write-Host "Clearing pip cache..." -ForegroundColor Yellow
pip cache purge 2>&1 | Out-Null
Write-Host "✓ Cache cleared" -ForegroundColor Green

Write-Host ""

# Try installation method 1: Direct with prefer-binary
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Method 1: Installing with pre-built wheels" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$method1Success = $false
try {
    Write-Host "Installing packages (this may take 5-10 minutes)..." -ForegroundColor Yellow
    pip install -r requirements.txt --prefer-binary --no-cache-dir
    
    if ($LASTEXITCODE -eq 0) {
        $method1Success = $true
        Write-Host "✓ Installation successful!" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Method 1 failed, trying alternative method..." -ForegroundColor Red
}

Write-Host ""

# If method 1 failed, try step-by-step installation
if (-not $method1Success) {
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "   Method 2: Step-by-step installation" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Stage 1: Core dependencies
    Write-Host "[1/6] Installing core dependencies..." -ForegroundColor Yellow
    pip install numpy==1.26.4 --prefer-binary --quiet
    pip install pydantic pyyaml python-dotenv requests tqdm pandas --prefer-binary --quiet
    Write-Host "✓ Core dependencies installed" -ForegroundColor Green
    
    # Stage 2: Sentence transformers
    Write-Host "[2/6] Installing sentence-transformers..." -ForegroundColor Yellow
    pip install sentence-transformers --prefer-binary --quiet
    Write-Host "✓ sentence-transformers installed" -ForegroundColor Green
    
    # Stage 3: FAISS
    Write-Host "[3/6] Installing FAISS..." -ForegroundColor Yellow
    pip install faiss-cpu --prefer-binary --quiet
    Write-Host "✓ FAISS installed" -ForegroundColor Green
    
    # Stage 4: LLM libraries
    Write-Host "[4/6] Installing LLM libraries..." -ForegroundColor Yellow
    pip install groq openai tiktoken --prefer-binary --quiet
    Write-Host "✓ LLM libraries installed" -ForegroundColor Green
    
    # Stage 5: PDF parsers
    Write-Host "[5/6] Installing PDF parsers..." -ForegroundColor Yellow
    pip install pymupdf pypdf2 --prefer-binary --quiet
    # Try docling, but don't fail if it doesn't work
    Write-Host "    Installing docling (this may take a moment)..." -ForegroundColor Gray
    pip install docling --prefer-binary 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ PDF parsers installed (including docling)" -ForegroundColor Green
    } else {
        Write-Host "✓ PDF parsers installed (docling skipped, will use pymupdf)" -ForegroundColor Yellow
    }
    
    # Stage 6: Remaining packages
    Write-Host "[6/6] Installing remaining packages..." -ForegroundColor Yellow
    pip install langchain langchain-openai langchain-community --prefer-binary --quiet
    pip install pytest pytest-cov --prefer-binary --quiet
    Write-Host "✓ Remaining packages installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Verifying Installation" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Test imports
$packages = @(
    @{Name="numpy"; Import="import numpy; print(numpy.__version__)"},
    @{Name="pandas"; Import="import pandas"},
    @{Name="sentence_transformers"; Import="import sentence_transformers"},
    @{Name="faiss"; Import="import faiss"},
    @{Name="groq"; Import="import groq"},
    @{Name="openai"; Import="import openai"},
    @{Name="pymupdf"; Import="import pymupdf"}
)

$allSuccess = $true
foreach ($pkg in $packages) {
    try {
        python -c $pkg.Import 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $($pkg.Name) - OK" -ForegroundColor Green
        } else {
            Write-Host "✗ $($pkg.Name) - Failed" -ForegroundColor Red
            $allSuccess = $false
        }
    } catch {
        Write-Host "✗ $($pkg.Name) - Failed" -ForegroundColor Red
        $allSuccess = $false
    }
}

# Test docling separately (optional)
try {
    python -c "from docling.document_converter import DocumentConverter" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ docling - OK" -ForegroundColor Green
    } else {
        Write-Host "○ docling - Not installed (will use pymupdf)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "○ docling - Not installed (will use pymupdf)" -ForegroundColor Yellow
}

Write-Host ""

if ($allSuccess) {
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "   ✓ Installation Complete!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Run setup: python setup_env.py" -ForegroundColor White
    Write-Host "  2. Get Groq API key: https://console.groq.com" -ForegroundColor White
    Write-Host "  3. Run pipeline: python run.py --pipeline" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "================================================" -ForegroundColor Yellow
    Write-Host "   ⚠ Installation completed with some warnings" -ForegroundColor Yellow
    Write-Host "================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Some packages failed to install." -ForegroundColor Yellow
    Write-Host "You can still run the system with minimal features." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "See WINDOWS_INSTALL.md for troubleshooting steps." -ForegroundColor Cyan
    Write-Host ""
}

# Show installed packages
Write-Host "Installed packages:" -ForegroundColor Cyan
pip list | Select-String -Pattern "groq|openai|faiss|sentence|numpy|docling|pymupdf"

Write-Host ""
Write-Host "Installation log saved to: pip-install.log" -ForegroundColor Gray
