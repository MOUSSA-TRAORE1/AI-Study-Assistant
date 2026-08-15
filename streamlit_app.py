import streamlit as st              #for web interface
from dotenv import load_dotenv      # allow loading api key securely
load_dotenv(override=True)

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# This is the same code but with UI (streamlit)

# Adding better layout
st.set_page_config(layout="wide")
# Title
st.title("📄 AI Study Assistant")
#Adding sidebar
mode=st.sidebar.selectbox(
    "Choose Mode",
    ["Ask Questions","Summarize", "Quiz", "Study Notes"]
)
'''
    below I'll write the function for different modes that will be in the system
    them they'll be invoked in the code
'''
#Adding summary generation
def generate_summary(context, llm): 
    prompt=f"""
    summarize the following content clearly:
    {context}
    Use bullet points
    """
    return llm.invoke(prompt).content

#Adding quizz generator
def generate_quiz(context, llm):
    prompt=f"""
    Create 5 quiz questions with answer from this
    {context}
    Format:
    Q1:
    A:
    """
    return llm.invoke(prompt).content

#Adding note generator
def generate_note(context,llm):
    prompt=f"""
    Turn this into structured study notes:

    - headings
    - bullet points
    - key ideas
    {context}
    """
    return llm.invoke(prompt).content
    



# Upload PDF
uploaded_files = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True) 

#File processing 
if uploaded_files:
    documents = []   # a list to store all the extracted pages (with metadata as well)

    for file in uploaded_files:
        reader = PdfReader(file)

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""      # extract text from pdf page or none

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": i + 1,
                        "source": file.name
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
    #db = Chroma.from_documents(chunks, embeddings) commented this will use a better version below, with session
    if "db" not in st.session_state:
        st.session_state.db = Chroma.from_documents(chunks, embeddings)

    db = st.session_state.db
    #Retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    #LLM definition
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


    # User question
query = st.text_input("Ask something")

if query and mode == "Ask Questions":
    docs = retriever.invoke(query)

    context = "\n\n".join([
        f"(Page {doc.metadata['page']}) {doc.page_content}"
        for doc in docs
    ])

    with st.spinner("Thinking..."):
        response = llm.invoke(f"""
        Answer using ONLY this context:
        {context}

        Question: {query}
        """)

    st.subheader("📌 Answer")
    st.write(response.content)

    st.subheader("📚 Sources")
    for doc in docs:
        st.write(f"Page {doc.metadata['page']} - {doc.metadata['source']}")





# Adding loaading spinner, I'll put it inside buttons
#with st.spinner("Processing..."): 
    



#Doing different ction nased on differnt mode
if mode == "Summarize":
    if st.button("Generate Summary"):
        with st.spinner("Processing..."):
            docs = retriever.invoke("Give me a summary")
            context = "\n\n".join([doc.page_content for doc in docs])

            summary = generate_summary(context, llm)

            st.subheader("📝 Summary")
            st.write(summary)

if mode == "Quiz":
    if st.button("Generate Quiz"):
        with st.spinner("Processing..."):
            docs = retriever.invoke("important concepts")
            context = "\n\n".join([doc.page_content for doc in docs])

            quiz = generate_quiz(context, llm)

            st.subheader("🧪 Quiz")
            st.write(quiz)

if mode == "Study Notes":
    if st.button("Generate Notes"):
        with st.spinner("Processing..."):
            docs = retriever.invoke("important concepts")
            context = "\n\n".join([doc.page_content for doc in docs])

            notes = generate_note(context, llm)

            st.subheader("📘 Notes")
            st.write(notes)