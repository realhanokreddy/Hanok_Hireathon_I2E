"""
Quick setup script for environment configuration.
Helps users create and configure their env/.env file interactively.
"""
import os
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_info(text):
    """Print info message."""
    print(f"ℹ  {text}")


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def print_warning(text):
    """Print warning message."""
    print(f"⚠  {text}")


def get_input(prompt, default=None, required=False):
    """Get user input with optional default."""
    if default:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "
    
    while True:
        value = input(prompt_text).strip()
        
        if not value and default:
            return default
        
        if not value and required:
            print_warning("This field is required. Please try again.")
            continue
        
        return value


def main():
    """Main setup function."""
    print_header("NASA QA System - Environment Setup")
    
    # Get project root
    project_root = Path(__file__).parent
    env_dir = project_root / "env"
    env_file = env_dir / ".env"
    env_example = env_dir / ".env.example"
    
    # Check if env directory exists
    if not env_dir.exists():
        print_warning("env/ directory not found. Creating...")
        env_dir.mkdir(parents=True, exist_ok=True)
        print_success("Created env/ directory")
    
    # Check if .env already exists
    if env_file.exists():
        print_warning(f".env file already exists at: {env_file}")
        overwrite = input("Do you want to overwrite it? (yes/no) [no]: ").strip().lower()
        if overwrite not in ['yes', 'y']:
            print_info("Setup cancelled. Existing .env file preserved.")
            return
    
    print_info("Let's set up your environment configuration!")
    print_info("Press Enter to use default values shown in brackets.")
    
    # Collect configuration
    config = {}
    
    # LLM Provider
    print_header("1. LLM Provider Configuration")
    print_info("Choose your LLM provider:")
    print_info("  - groq: Free tier, fast, 60 req/min (recommended)")
    print_info("  - openai: Paid, high quality")
    config['LLM_PROVIDER'] = get_input("LLM Provider (groq/openai)", "groq")
    
    # Groq Configuration
    if config['LLM_PROVIDER'] == 'groq':
        print_header("2. Groq API Configuration")
        print_info("Get your free API key from: https://console.groq.com")
        config['GROQ_API_KEY'] = get_input("Groq API Key", required=True)
        
        print_info("Available models:")
        print_info("  - llama3-70b-8192 (best quality)")
        print_info("  - llama3-8b-8192 (faster)")
        print_info("  - mixtral-8x7b-32768 (32K context)")
        config['GROQ_MODEL'] = get_input("Groq Model", "llama3-70b-8192")
        config['GROQ_TEMPERATURE'] = get_input("Temperature", "0.1")
        config['GROQ_MAX_TOKENS'] = get_input("Max Tokens", "2000")
    
    # OpenAI Configuration
    if config['LLM_PROVIDER'] == 'openai':
        print_header("2. OpenAI API Configuration")
        print_info("Get your API key from: https://platform.openai.com/api-keys")
        config['OPENAI_API_KEY'] = get_input("OpenAI API Key", required=True)
        config['OPENAI_MODEL'] = get_input("OpenAI Model", "gpt-4-turbo-preview")
        config['OPENAI_TEMPERATURE'] = get_input("Temperature", "0.1")
        config['OPENAI_MAX_TOKENS'] = get_input("Max Tokens", "2000")
    
    # Embedding Provider
    print_header("3. Embedding Configuration")
    print_info("Choose your embedding provider:")
    print_info("  - local: Free, runs on your machine (recommended)")
    print_info("  - openai: Paid, higher quality")
    config['EMBEDDING_PROVIDER'] = get_input("Embedding Provider (local/openai)", "local")
    
    if config['EMBEDDING_PROVIDER'] == 'local':
        print_info("Local model: sentence-transformers/all-MiniLM-L6-v2")
        config['LOCAL_EMBEDDING_MODEL'] = "sentence-transformers/all-MiniLM-L6-v2"
        config['LOCAL_EMBEDDING_DIMENSION'] = "384"
        config['VECTOR_STORE_DIMENSION'] = "384"
    else:
        config['OPENAI_EMBEDDING_MODEL'] = "text-embedding-3-large"
        config['OPENAI_EMBEDDING_DIMENSION'] = "3072"
        config['VECTOR_STORE_DIMENSION'] = "3072"
    
    # Other configurations (use defaults)
    print_header("4. Using Default Settings")
    print_info("The following will use recommended default values:")
    print_info("  - Chunk size: 800 tokens")
    print_info("  - Retrieval: Hybrid (vector + keyword)")
    print_info("  - Multi-hop: Enabled (2 hops)")
    print_info("  - Vector store: FAISS")
    
    # Write .env file
    print_header("5. Creating .env File")
    
    try:
        # Read example file to get all variables
        if env_example.exists():
            with open(env_example, 'r') as f:
                lines = f.readlines()
        else:
            print_warning(".env.example not found. Creating minimal .env file.")
            lines = []
        
        # Write .env file
        with open(env_file, 'w') as f:
            for line in lines:
                # Replace configured values
                if '=' in line and not line.strip().startswith('#'):
                    key = line.split('=')[0]
                    if key in config:
                        f.write(f"{key}={config[key]}\n")
                    else:
                        f.write(line)
                else:
                    f.write(line)
        
        print_success(f"Created .env file at: {env_file}")
        
        # Show summary
        print_header("Setup Complete!")
        print("\nConfiguration Summary:")
        print(f"  LLM Provider: {config.get('LLM_PROVIDER', 'N/A')}")
        print(f"  Embedding Provider: {config.get('EMBEDDING_PROVIDER', 'N/A')}")
        print(f"  Vector Dimension: {config.get('VECTOR_STORE_DIMENSION', 'N/A')}")
        
        print("\nNext Steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run the pipeline: python run.py --pipeline")
        print("  3. Ask questions: python run.py --interactive")
        
        print_info(f"\nYou can edit {env_file} anytime to change settings.")
        
    except Exception as e:
        print_warning(f"Error creating .env file: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
