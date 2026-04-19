import os
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader
import sys

# Allow importing from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

def load_documents(data_dir: str = DATA_DIR) -> List[Any]:
    """
    Load PDF documents, specifically targeting the textbook.
    Returns: List of LangChain Documents containing page_content and metadata (source, page).
    """
    data_path = Path(data_dir).resolve()
    print(f"[INFO] Scanning for textbooks in: {data_path}")
    documents = []

    # Target PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[INFO] Found {len(pdf_files)} textbook file(s).")
    
    for pdf_file in pdf_files:
        print(f"[INFO] Ingesting: {pdf_file.name}")
        try:
            # PyMuPDFLoader automatically adds 'source' and 'page' metadata
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[SUCCESS] Extracted {len(loaded)} pages from {pdf_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to extract from {pdf_file.name}: {e}")

    print(f"[INFO] Total pages (documents) ready for chunking: {len(documents)}")
    return documents

if __name__ == "__main__":
    docs = load_documents()
    if docs:
        print("\n--- Metadata Example from Page 1 ---")
        print(docs[0].metadata)