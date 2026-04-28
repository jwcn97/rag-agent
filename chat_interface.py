import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def get_collection_name() -> str:
    return os.getenv("COLLECTION_NAME", "index").strip()


def get_persist_directory() -> str:
    return os.getenv("PERSIST_DIRECTORY", "./db/index").strip()


def ensure_persist_directory_writable() -> None:
    persist_path = Path(get_persist_directory())
    persist_path.mkdir(parents=True, exist_ok=True)
    probe_file = persist_path / ".write_test"
    probe_file.write_text("ok", encoding="utf-8")
    probe_file.unlink(missing_ok=True)


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))


@st.cache_resource
def get_vector_store():
    ensure_persist_directory_writable()
    return Chroma(
        collection_name=get_collection_name(),
        embedding_function=get_embeddings(),
        persist_directory=get_persist_directory(),
    )


def get_retriever():
    return get_vector_store().as_retriever(search_kwargs={"k": 4})


@st.cache_resource
def get_chat_model():
    return init_chat_model(os.getenv("CHAT_MODEL"))


def build_prompt(question: str, context: str) -> str:
    return f"""You are a helpful assistant answering questions based on the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""


def ingest_uploaded_pdfs(uploaded_files) -> tuple[int, list[str]]:
    all_chunks = []
    errors = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    upload_tmp_dir = Path("docs/.tmp_uploads")
    upload_tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for uploaded_file in uploaded_files:
            try:
                temp_pdf_path = upload_tmp_dir / uploaded_file.name
                temp_pdf_path.write_bytes(uploaded_file.getvalue())

                loader = PyPDFLoader(str(temp_pdf_path))
                data = loader.load()
                chunks = splitter.split_documents(data)

                for chunk in chunks:
                    chunk.metadata["uploaded_filename"] = uploaded_file.name

                all_chunks.extend(chunks)
            except Exception as exc:
                errors.append(f"{uploaded_file.name}: {exc}")
    finally:
        shutil.rmtree(upload_tmp_dir, ignore_errors=True)

    if all_chunks:
        get_vector_store().add_documents(documents=all_chunks)

    return len(all_chunks), errors


def delete_whole_db():
    persist_directory = get_persist_directory()
    if persist_directory and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    Path(persist_directory).mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="FAQ Chat", page_icon="💬")
st.title("💬 Ask Your Agent")
st.caption(
    f"Retrieval-Augmented Q&A over your data from `{get_collection_name()}`"
)

with st.sidebar:
    st.header("Data Management")

    uploaded_files = st.file_uploader(
        "Load files (drag and drop PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Load files", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner("Indexing documents into Chroma..."):
                chunk_count, errors = ingest_uploaded_pdfs(uploaded_files)
            st.cache_resource.clear()
            st.success(f"Loaded {len(uploaded_files)} files ({chunk_count} chunks).")
            if errors:
                st.error("Some files failed to load:")
                for err in errors:
                    st.write(f"- {err}")
            st.rerun()

    st.divider()
    st.subheader("Delete whole DB")
    confirm_delete = st.text_input("Type DELETE to confirm", placeholder="DELETE")

    if st.button("Delete whole DB", type="primary", use_container_width=True):
        if confirm_delete != "DELETE":
            st.error("Confirmation text mismatch. Type DELETE to proceed.")
        else:
            delete_whole_db()
            st.cache_resource.clear()
            st.success("Database deleted.")
            st.rerun()

query = st.text_input("Ask a question about your indexed documents:")

if query:
    retriever = get_retriever()
    model = get_chat_model()
    docs = retriever.invoke(query)

    if not docs:
        st.warning("No matching chunks found.")
    else:
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = build_prompt(query, context)
        response = model.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved chunks"):
            for idx, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "unknown")
                st.markdown(f"**Chunk {idx}** — source: `{source}`, page: `{page}`")
                st.write(doc.page_content)
                st.divider()
