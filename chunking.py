import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

# CONFIG
FOLDER_PATH = "./Pdfs"
CHUNK_SIZE = 300      # in tokens (approximate, since we're using tiktoken as an estimate)
CHUNK_OVERLAP = 50    # in tokens

# 1. Initialize the standard client (Make sure GEMINI_API_KEY is in your environment variables)
client = genai.Client()

# 2. Update your count_tokens function to use the client
def count_tokens(text):
    """Returns an approximate number of tokens a given text will become."""
    # This calls the fast, built-in count_tokens API method
    response = client.models.count_tokens(
        model='gemini-2.5-flash',
        contents=text
    )
    return response.total_tokens

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
    print(pdf_docs)

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