import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Centered compact layout
st.set_page_config(page_title="Public Policy RAG Demo: NCCI Manual Retrieval", layout="centered")

st.markdown("""
<style>
    .block-container {max-width: 900px; padding-top: 1rem; padding-bottom: 3rem;}
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; text-align: center;}
    h3 {color: #34495e;}
    .stMarkdown {font-size: 1.1rem; line-height: 1.6;}
</style>
""", unsafe_allow_html=True)

st.title("Public Policy RAG Demo: Medicaid NCCI Manual Retrieval (Personal Project)")

st.markdown("""
### About This Demo
This is a **personal open-source project** exploring retrieval-augmented generation (RAG) for policy and compliance research.  
All answers are grounded **exclusively** in the official public **2025 CMS Medicaid National Correct Coding Initiative (NCCI) Policy Manual**—a freely downloadable document from cms.gov (direct link: [2025 Medicaid NCCI Policy Manual](https://www.cms.gov/files/document/2025nccimedicaidpolicymanualcomplete.pdf)).

**No private, confidential, or personal data is used**—purely public federal guidance available to anyone. Not affiliated with or endorsed by any government agency.

#### Why I Built This
Professionals working with federal healthcare policy (e.g., state compliance officers, analysts, or consultants) often need to quickly locate and understand specific rules in lengthy official documents—like coding edits, guidelines, or implementation details—to support accurate decision-making or rate adjustments.  
This prototype shows how grounded AI can help by retrieving relevant, cited excerpts in seconds, making complex public documents easier to navigate.

**Intended Value**: Faster access to precise information from official sources, aiding general research, policy review, or professional workflows—while ensuring responses stay strictly tied to the original text for reliability.

#### How to Use 🔍
Ask natural-language questions about NCCI rules. Responses provide factual summaries + expandable source excerpts with page numbers from the manual.

**Try these examples** (copy-paste):
- "What is the National Correct Coding Initiative?"
- "Define Medically Unlikely Edits (MUEs) and their purpose."
- "How do Procedure-to-Procedure (PTP) edits work?"
- "Explain the role of modifiers in NCCI edits."

Built with LangChain + FAISS + Google Gemini + Streamlit—open-source with cloud LLM for demo.
""")

@st.cache_resource
def load_and_index_documents():
    index_path = "faiss_index"
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={"k": 10})
    
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    
    expected_pdf = "cms_ncci_2025_policy_manual.pdf"  # Update if needed
    if len(pdf_files) != 1 or expected_pdf not in pdf_files:
        st.error(f"Expected only '{expected_pdf}' in folder. Found: {pdf_files if pdf_files else 'none'}.")
        st.stop()
    
    st.info(f"Processing {pdf_files[0]}...")
    
    loader = PyPDFLoader(pdf_files[0])
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    splits = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(index_path)
    st.success("Index ready!")
    
    return vectorstore.as_retriever(search_kwargs={"k": 10})

retriever = load_and_index_documents()

# Gemini LLM (free tier)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.05)

# Prompt (same)
prompt = PromptTemplate.from_template("""
You are an expert on CMS Medicaid NCCI policy. Answer the question using only the provided context excerpts from the 2025 manual. 
If the context doesn't cover it, say "Not covered in the provided manual excerpts."
Cite page numbers where possible.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 'unknown') + 1
        formatted.append(f"Excerpt {i} (Page {page}):\n{doc.page_content}\n")
    return "\n".join(formatted)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat (same)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Retrieved excerpts"):
                st.markdown(message["sources"])

if prompt := st.chat_input("Ask about NCCI policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating..."):
            response = chain.invoke(prompt)
            sources = format_docs(retriever.invoke(prompt))
            st.markdown(response)
            if sources:
                with st.expander("Retrieved excerpts"):
                    st.markdown(sources)
                st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})

st.caption("Personal open-source project—feedback welcome! Public document only, no affiliation.")