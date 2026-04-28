import os
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF
loader = PyPDFLoader("./docs/xxx.pdf")
data = loader.load()

# 2. Split into 1000-character chunks with 200-character overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)

print(f"Split into {len(chunks)} text chunks.")

model = init_chat_model(os.getenv("CHAT_MODEL"))
embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
vector_store = Chroma(
  collection_name=os.getenv("COLLECTION_NAME"),
  embedding_function=embeddings,
  persist_directory=os.getenv("PERSIST_DIRECTORY"),
)

_ = vector_store.add_documents(documents=chunks)