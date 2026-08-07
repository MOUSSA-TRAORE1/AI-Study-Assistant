# This is the same code but with UI (streamlit)

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Title
st.title("📄 AI Study Assistant")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Read PDF
    reader = PdfReader(uploaded_file)
    
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page": i + 1,
                    "source": uploaded_file.name
                }
            )
        )

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    db = Chroma.from_documents(chunks, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # User question
    query = st.text_input("Ask a question about your document")

    if query:
        docs = retriever.invoke(query)

        context = "\n\n".join([
            f"(Page {doc.metadata['page']}) {doc.page_content}"
            for doc in docs
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        response = llm.invoke(f"""
        Answer the question using ONLY the context below.
        Context:
        {context}

        Question:
        {query}
        """)

        st.subheader("📌 Answer")
        st.write(response.content)

        st.subheader("📚 Sources")
        for doc in docs:
            st.write(f"Page {doc.metadata['page']} - {doc.metadata['source']}")