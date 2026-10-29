import os
import streamlit as st

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="College AI Assistant", page_icon="🎓", layout="wide")
st.title("🎓 College AI Assistant (RAG)")
st.caption("Official Student Query Assistant with Source Tracking")

# --- UI: Sidebar Settings (Uploader Removed) ---
st.sidebar.header("⚙️ RAG Settings")
chunk_size = st.sidebar.slider("Chunk Size", 300, 1000, 500, step=50)
top_k = st.sidebar.slider("Retrieved Chunks (k)", 1, 8, 4)

# Directory where your 5 test documents are saved
DOCUMENTS_DIR = "./documents"

@st.cache_resource(show_spinner=False)
def process_and_index_documents(directory, c_size):
    all_docs = []
    loaded_files = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None, 0, 0, []

    # Automatically read all files from the local folder
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        suffix = os.path.splitext(filename)[1].lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            elif suffix == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue # Skip unsupported file types
            
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            
            all_docs.extend(docs)
            loaded_files.append(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not all_docs:
        return None, 0, 0, []

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size, 
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db, len(loaded_files), len(chunks), loaded_files

# --- Initialize FAISS Automatically ---
vector_db = None
with st.spinner("Scanning local documents folder and building FAISS index..."):
    vector_db, total_docs, total_chunks, loaded_files = process_and_index_documents(DOCUMENTS_DIR, chunk_size)

if vector_db:
    st.sidebar.success(f"✅ Successfully loaded {total_docs} files from local folder.")
    with st.sidebar.expander("📄 View Loaded Files"):
        for file in loaded_files:
            st.markdown(f"- `{file}`")
else:
    st.error(f"No documents found! Please place your .txt or .pdf files inside the `{DOCUMENTS_DIR}` folder.")

# --- RAG Setup ---
llm = ChatOllama(model="gemma2:2b", temperature=0)

system_prompt = """You are an official College AI Assistant answering student questions based ONLY on the provided context.

Strict Rules:
1. Base your answer strictly on the provided Context.
2. If the answer is not in the Context, explicitly state: "This information is not available in the provided college documents." Do not invent facts.
3. If context from different sources presents conflicting information, point out the contradiction and cite both conflicting sources.
4. Synthesize details if information comes from multiple sources.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# --- User Query ---
question = st.text_input("Ask a question regarding college policies, fees, or rules:")

if question:
    if not vector_db:
        st.error("Cannot answer questions without documents in the knowledge base.")
    else:
        with st.spinner("Searching knowledge base..."):
            retrieved_docs = vector_db.similarity_search(question, k=top_k)

            formatted_context = "\n\n".join([
                f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in retrieved_docs
            ])

            response = chain.invoke({
                "context": formatted_context,
                "question": question
            })

        st.markdown("### 💬 Answer")
        st.write(response)

        st.markdown("---")
        with st.expander("🔍 View Retrieved Sources"):
            for idx, doc in enumerate(retrieved_docs, start=1):
                src = doc.metadata.get("source", "Unknown")
                st.markdown(f"**Source {idx}:** `{src}`")
                st.info(doc.page_content)