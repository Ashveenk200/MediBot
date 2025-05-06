import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Set Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

# Optional: Log in to Hugging Face Hub
from huggingface_hub import login
login(token=hf_token)

# Step 1: Load PDF files
DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print("Length of documents:", len(documents))

# Step 2: Split into chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

text_chunks = create_chunks(extracted_data=documents)
print("Length of chunks:", len(text_chunks))

# Step 3: Create vector embedding
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # Pass the token if required by the model
        model_kwargs={"token": hf_token}
    )
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Connect with FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)