import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# CONFIG
FOLDER_PATH = "./Pdfs"
CHUNK_SIZE = 300      # in tokens
CHUNK_OVERLAP = 50    # in tokens

# Local fast tokenizer (runs in memory, zero API calls)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Returns local token count instantly without network calls."""
    return len(tokenizer.encode(text))

# DOCUMENT LOADING
def read_documents(folder_path):
    """Loads all .txt files from a folder into LangChain Document objects."""

    txt_loader = DirectoryLoader(
        path=folder_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    txt_docs = txt_loader.load()

    pdf_loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        loader_kwargs={
            "extract_tables": "markdown",  # Formats tables into clear markdown
            "extract_images": True  # Extracts text from images inside the PDF
        }
    )
    pdf_docs = pdf_loader.load()
    return pdf_docs + txt_docs

# CHUNKING
def split_documents(documents, chunk_size, chunk_overlap):
    """Splits documents into token-measured chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        is_separator_regex=False,
        separators=["\n\n", "\n", "\n|", " ", ""]
    )
    return text_splitter.split_documents(documents)