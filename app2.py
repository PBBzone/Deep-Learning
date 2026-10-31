import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.title("RAG Chatbot with LLM")

@st.cache_resource
def load_vectorstore():
    # Load Document
    loader = TextLoader("CPP.txt", encoding="utf-8")
    documents = loader.load()

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    final_documents = text_splitter.split_documents(documents)

    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(final_documents, embeddings)
    return db

db = load_vectorstore()

llm = Ollama(model="gemma2:9b")


user_query = st.text_input("Enter your question here")

if user_query:
    with st.spinner("Searching for relevant context..."):
        docs = db.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"Answer the following question based on the context provided:\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"

        response = llm.invoke(prompt)

        st.subheader("Answer:")
        st.write(response)