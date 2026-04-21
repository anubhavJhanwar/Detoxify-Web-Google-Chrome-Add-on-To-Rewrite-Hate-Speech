"""
setup.py
--------
One-time setup script: installs dependencies and downloads NLP models.
Run this before starting the server.

Usage: python setup.py
"""

import subprocess
import sys


def run(cmd, desc):
    print(f"\n>>> {desc}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"WARNING: Command failed: {cmd}")
    else:
        print(f"✓ Done")


def main():
    print("=" * 60)
    print("ToxiClear — Setup")
    print("=" * 60)

    # Install Python dependencies
    run(
        f"{sys.executable} -m pip install -r backend/requirements.txt",
        "Installing Python dependencies"
    )

    # Download spaCy model
    run(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy English model (en_core_web_sm)"
    )

    # Download NLTK data
    run(
        f'{sys.executable} -c "import nltk; nltk.download(\'stopwords\'); nltk.download(\'vader_lexicon\'); nltk.download(\'punkt\')"',
        "Downloading NLTK data (stopwords, vader_lexicon, punkt)"
    )

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("\nNext steps:")
    print("  1. (Optional) Add datasets to data/ folder — see data/README.md")
    print("  2. Train the model:")
    print("       cd backend && python train_model.py")
    print("       OR with Jigsaw data:")
    print("       cd backend && python train_model.py ../data/jigsaw_dataset.csv")
    print("  3. Start the API server:")
    print("       cd backend && uvicorn app:app --reload --port 8000")
    print("  4. Open frontend/index.html in your browser")
    print("  5. Load extension/ as unpacked extension in Chrome")
    print("=" * 60)


if __name__ == "__main__":
    main()
