import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
from datetime import datetime
from tavily import TavilyClient
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os, shutil

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Uploading my pdf file
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Splitting texts to chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    return text_splitter.split_text(text)


# Create FAISS Vectorstore
def get_vector_store(text_chunks):
    # clear old index
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say
    "Answer is not available in the pdf context".
    Don't try to make up an answer.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Tools
def calc(query: str) -> str:
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error: {e}"

def today_date(query: str) -> str:
    return str(datetime.today().date())

def web_search(query: str) -> str:
    results = tavily.search(query)
    return str(results)

def pdf_qa(query: str) -> str:
    if not os.path.exists("faiss_index"):
        return "No PDF uploaded or processed yet."

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(query, k=2)
    if not docs:
        return "No relevant info found in PDF."

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return response["output_text"]


# AI Agent
tools = [
    Tool(name="Calculator", func=calc, description="Useful for doing math calculations."),
    Tool(name="Today's Date", func=today_date, description="Useful for telling today's date."),
    Tool(name="Web Search", func=web_search, description="Useful for answering general knowledge or current events."),
    Tool(name="PDF QA",func=pdf_qa,description=(
        "Always use this tool if the user asks about content from the uploaded PDF. "
        "If the answer is not found in the PDF, the tool will respond accordingly. "
        "Use Web Search only if the information is clearly not in the PDF."))
]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


# Streamlit for UI in output
def main():
    st.set_page_config("RAG")
    st.header("Search Answers")

    with st.sidebar:
        st.title("File:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Submit & Process",
            accept_multiple_files=True
        )
        if st.button("Submit & Process", key="process_pdf"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF processed")
            else:
                st.warning("Please upload at least one PDF")



    user_question = st.text_input("Ask a Question", key="input")

    if user_question.strip():
        with st.spinner("Thinking..."):
            result = agent.run(user_question)
            st.subheader("Agent Answer:")
            st.write(result)

if __name__ == "__main__":
    main()
