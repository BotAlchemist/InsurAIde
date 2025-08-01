import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from functools import partial

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Load documents
utf8_loader = partial(TextLoader, encoding="utf-8")
loader = DirectoryLoader("documents", glob="*.txt", loader_cls=utf8_loader)
#loader = DirectoryLoader("documents", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Create/Open vector DB
persist_dir = "chromadb_store"
embeddings = OpenAIEmbeddings(api_key=openai_key)

if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
else:
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-4o", api_key=openai_key, temperature=0.2)

# === Initialize session state ===
if "expanded_requirement" not in st.session_state:
    st.session_state["expanded_requirement"] = ""
if "editable_requirement" not in st.session_state:
    st.session_state["editable_requirement"] = ""
if "relevant_docs" not in st.session_state:
    st.session_state["relevant_docs"] = []

# === Tabs Layout ===
st.title("AI-Powered Insurance Requirement Expander")
tab1, tab2 = st.tabs(["🔍 Expand Requirement", "🧪 Generate Test Cases"])

# ---------------- TAB 1: Expand Requirement ----------------
with tab1:
    user_query = st.text_area("Enter a one-line requirement or issue")

    if st.button("Expand with Context"):
        if user_query:
            with st.spinner("Expanding with knowledge base..."):
                relevant_docs = vectordb.similarity_search(user_query, k=4)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                prompt = f"""You are a business analyst assistant. Based on the following context, expand the given one-line requirement into a detailed business requirement or user story.

Context:
{context}

Requirement:
{user_query}

Expanded Requirement:"""

                response = llm.invoke(prompt)
                expanded_text = response.content if hasattr(response, "content") else response

            #st.success("Expanded Requirement")

            # Save to session state
            st.session_state["expanded_requirement"] = expanded_text
            st.session_state["editable_requirement"] = expanded_text
            st.session_state["relevant_docs"] = relevant_docs

    # Show editable text area if expansion exists
    if st.session_state["editable_requirement"]:
        st.text_area(
            "Review or Edit Expanded Requirement Below:",
            value=st.session_state["editable_requirement"],
            key="editable_requirement",
            height=500
        )

        with st.expander("References"):
            for i, doc in enumerate(st.session_state["relevant_docs"], start=1):
                st.markdown(f"**{i}. Source:** `{doc.metadata.get('source', 'Unknown')}`")
                st.markdown(f"> {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")

# ---------------- TAB 2: Generate Test Cases ----------------
with tab2:
    st.markdown("### Step 2: Generate Functional Test Cases")

    if st.button("Generate Functional Test Cases"):
        edited_input = st.session_state.get("editable_requirement", "")
        if edited_input:
            with st.spinner("Generating test cases..."):
                test_case_prompt = f"""You are a QA analyst. Based on the following detailed business requirement, generate 5  functional test cases. Each test case should include:
- Test Case Title
- Input
- Action
- Expected Result

Business Requirement:
{edited_input}
"""
                test_response = llm.invoke(test_case_prompt)
                #st.success("Functional Test Cases")
                st.write(test_response.content if hasattr(test_response, "content") else test_response)
        else:
            st.warning("No expanded requirement found. Please use the first tab to generate one.")

