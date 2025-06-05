import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from openai import OpenAI
import tempfile

# 👇 Secure OpenAI key from Streamlit Secrets
client = OpenAI(api_key=st.secrets["openai_api_key"])
embedding_model = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])

st.title("📊 InsightGPT: Ask Questions About Your CSV")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Save uploaded CSV to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load CSV as document
    loader = CSVLoader(file_path=tmp_path)
    documents = loader.load()

    # Build FAISS vector database
    db = FAISS.from_documents(documents, embedding_model)

    # Use ChatOpenAI with the secured key
    llm = ChatOpenAI(
        openai_api_key=st.secrets["openai_api_key"],
        temperature=0,
        model_name="gpt-3.5-turbo"
    )

    # Create Retrieval QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    # Input from user
    query = st.text_input("Ask a question about your data:")

    if query:
        response = qa.run(query)
        st.subheader("🧠 GPT Answer:")
        st.write(response)
