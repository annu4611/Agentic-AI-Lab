import os
import warnings
from dotenv import load_dotenv

# Suppress known deprecation warnings from legacy modules
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-community.*")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

INDEX_DIR = "faiss_index"


def get_vectorstore(pdf_path: str, index_dir: str = None, force_rebuild: bool = False):
    """Loads PDF, splits into chunks, and creates or loads a cached FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Determine unique cache dir based on filename or explicit directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if index_dir is None:
        file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        # Keep default directory if default elevate.pdf or generic policy
        if "elevate" in file_stem.lower():
            index_dir = os.path.join(base_dir, "faiss_index")
        else:
            index_dir = os.path.join(base_dir, "faiss_index", file_stem)
    elif not os.path.isabs(index_dir):
        index_dir = os.path.join(base_dir, index_dir)

    # Fallback to base faiss_index if candidate index doesn't exist but default faiss_index does
    if not (os.path.exists(os.path.join(index_dir, "index.faiss"))):
        default_idx = os.path.join(base_dir, "faiss_index")
        if os.path.exists(os.path.join(default_idx, "index.faiss")):
            index_dir = default_idx

    # Reuse vector store if already built and not forced
    if os.path.exists(os.path.join(index_dir, "index.faiss")) and not force_rebuild:
        print(f"Loading existing vector database from '{index_dir}'...")
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    print(f"Reading PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split into chunks (larger chunks = fewer embedding calls)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.split_documents(pages)
    print(f"Created {len(docs)} chunks from {len(pages)} pages. Generating embeddings...")

    # Build FAISS vector store and save locally so it only embeds once
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"Vector database saved to '{index_dir}'!")
    return vectorstore


def format_docs(docs) -> str:
    """Formats retrieved Document chunks into a combined context string."""
    return "\n\n".join(doc.page_content for doc in docs)
