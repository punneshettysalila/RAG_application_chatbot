# rag_pipeline.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import tempfile


def load_and_index_pdfs(uploaded_files):
    """Load PDFs, split into chunks, embed and return FAISS vector store."""
    all_docs = []

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        # Save to temp file because PyPDFLoader needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)
        os.unlink(tmp_path)  # Clean up temp file

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDF files.")

    # Embed using a lightweight sentence transformer
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_context(vectorstore, query, k=4):
    """Retrieve top-k relevant chunks for a query."""
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

