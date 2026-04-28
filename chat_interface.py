import os
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings


load_dotenv()


@st.cache_resource
def get_retriever():
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    vector_store = Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        embedding_function=embeddings,
        persist_directory=os.getenv("PERSIST_DIRECTORY"),
    )
    return vector_store.as_retriever(search_kwargs={"k": 4})


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


st.set_page_config(page_title="FAQ Chat", page_icon="💬")
st.title("💬 Ask Your Agent")
st.caption(
    f"Retrieval-Augmented Q&A over your data from `{os.getenv('COLLECTION_NAME', 'db')}`"
)

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
