import os
import shutil
import time
from langchain_google_genai import ChatGoogleGenerativeAI
import git
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Codebase Explainer",
    page_icon="🔍",
    layout="wide"
)

# ── Initialize session state ──────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.history     = ChatMessageHistory()
    st.session_state.messages    = []
    st.session_state.repo_name   = ""
    st.session_state.indexed     = False
    st.session_state.stats       = {}

# ── Load models ───────────────────────────────────────────
@st.cache_resource
@st.cache_resource
def load_models():
    # Try Groq first — fastest
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=500
        )
        # Test if it works
        llm.invoke("hi")
        print("Using Groq")
    except Exception:
        # Fallback to Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_output_tokens=500
        )
        print("Using Gemini fallback")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return llm, embeddings

llm, embeddings = load_models()
parser = StrOutputParser()

# ── Core functions ────────────────────────────────────────
def clone_repo(github_url):
    repo_name  = github_url.rstrip("/").split("/")[-1]
    clone_path = f"cloned_repos/{repo_name}"
    if os.path.exists(clone_path):
        shutil.rmtree(clone_path)
    os.makedirs("cloned_repos", exist_ok=True)
    git.Repo.clone_from(github_url, clone_path)
    return clone_path, repo_name

def load_code_files(repo_path):
    extensions = ["py", "js", "ts", "md", "txt", "json", "css", "html"]
    all_docs   = []
    for ext in extensions:
        try:
            loader = DirectoryLoader(
                repo_path,
                glob=f"**/*.{ext}",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                silent_errors=True
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["file_name"] = os.path.basename(
                    doc.metadata.get("source", "unknown")
                )
                doc.metadata["file_type"] = ext
            all_docs.extend(docs)
        except Exception:
            continue
    return all_docs

def split_and_index(all_docs):
    from langchain_text_splitters import Language

    EXTENSION_TO_LANGUAGE = {
        "py": Language.PYTHON,
        "js": Language.JS,
        "ts": Language.TS,
        "jsx": Language.JS,
        "tsx": Language.TS,
        "java": Language.JAVA,
        "cpp": Language.CPP,
        "c": Language.CPP,
        "go": Language.GO,
        "rb": Language.RUBY,
        "rs": Language.RUST,
        "md": Language.MARKDOWN,
    }

    all_chunks = []
    for doc in all_docs:
        ext = doc.metadata.get("file_type", "").lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)
        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=2000,
                chunk_overlap=300
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
        all_chunks.extend(splitter.split_documents([doc]))

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )
    return vectorstore, len(all_docs), len(all_chunks)


def ask_question(question, vectorstore, history):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
    )
    docs = retriever.invoke(question)
    context = "\n\n".join([
        f"# File: {d.metadata['file_name']}\n{d.page_content}"
        for d in docs
    ])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert code analyst for a GitHub repository.\n"
         "Answer questions using the retrieved code chunks below.\n\n"
         "Rules:\n"
         "- Always name the exact file where you found the answer\n"
         "- Prioritize source code files (.py, .js, .ts) over documentation (README, conf.py, setup.py)\n"
         "- If implementation is spread across files, piece it together\n"
         "- If you see a method name or partial logic, explain what it does\n"
         "- NEVER say 'not in codebase' if you found related code or methods\n"
         "- Give specific details: method names, parameters, logic flow\n"
         "- If truly nothing relevant exists, say what you DID find instead\n\n"
         "Code context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    chain = prompt | llm | parser
   
    for attempt in range(3):
                try:
                    time.sleep(0.5)
                    response = chain.invoke({
                        "context": context,
                        "history": history.messages,
                        "question": question
                    })
                    history.add_user_message(question)
                    history.add_ai_message(response)
                    return response
                except Exception as e:
                    err = str(e).lower()
                    if "429" in err or "rate limit" in err:
                        if attempt < 2:
                            time.sleep(10 * (attempt + 1))
                            continue
                        return "⚠️ Rate limit hit. Resets midnight UTC."
                    elif "401" in err or "invalid api key" in err:
                        return "⚠️ Invalid API key. Update GROQ_API_KEY in .env"
                    elif "timeout" in err or "connection" in err:
                        if attempt < 2:
                            time.sleep(5)
                            continue
                        return "⚠️ Connection timed out. Try again."
                    else:
                        return f"⚠️ Error: {str(e)}"

# ── UI ────────────────────────────────────────────────────
st.title("🔍 AI Codebase Explainer")
st.markdown(
    "Paste any **public GitHub repo URL** — "
    "ask questions about the code in plain English"
)
st.divider()

with st.sidebar:
    st.header("📦 Load Repository")

    # ── Quick fill buttons ────────────────────────────────
    st.markdown("**Try these:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Spoon-Knife", use_container_width=True):
            st.session_state["prefill_url"] = "https://github.com/octocat/Spoon-Knife"
    with col2:
        if st.button("📁 Flask", use_container_width=True):
            st.session_state["prefill_url"] = "https://github.com/pallets/flask"

    # ── URL input ─────────────────────────────────────────
    default_url = st.session_state.get("prefill_url", "")
    github_url  = st.text_input(
        "GitHub Repository URL",
        value=default_url,
        placeholder="https://github.com/username/repo"
    )

    # ── Load button ───────────────────────────────────────
    if github_url:
        if st.button(
            "🚀 Load & Index",
            use_container_width=True,
            type="primary"
        ):
            try:
                st.session_state.messages = []
                st.session_state.history  = ChatMessageHistory()
                st.session_state.indexed  = False

                with st.spinner("Step 1/3: Cloning repository..."):
                    clone_path, repo_name = clone_repo(github_url)

                with st.spinner("Step 2/3: Loading files..."):
                    all_docs = load_code_files(clone_path)
                    if not all_docs:
                        st.error("No readable files found!")
                        st.stop()

                with st.spinner(f"Step 3/3: Indexing {len(all_docs)} files..."):
                    vectorstore, n_files, n_chunks = split_and_index(all_docs)

                st.session_state.vectorstore = vectorstore
                st.session_state.repo_name   = repo_name
                st.session_state.indexed     = True
                st.session_state.stats       = {
                    "files" : n_files,
                    "chunks": n_chunks
                }
                # Clear prefill after successful load
                if "prefill_url" in st.session_state:
                    del st.session_state["prefill_url"]
                st.success("✅ Ready!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.indexed:
        st.divider()
        st.metric("Files",  st.session_state.stats["files"])
        st.metric("Chunks", st.session_state.stats["chunks"])
        st.markdown(f"**Repo:** {st.session_state.repo_name}")

        if st.button("🔄 New Repo", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.indexed     = False
            st.session_state.messages    = []
            st.session_state.history     = ChatMessageHistory()
            if "prefill_url" in st.session_state:
                del st.session_state["prefill_url"]
            st.rerun()

# ── Main area ─────────────────────────────────────────────
if not st.session_state.indexed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nPaste GitHub URL")
    with col2:
        st.info("**Step 2**\nClick Load & Index")
    with col3:
        st.info("**Step 3**\nAsk questions")

    st.divider()
    st.markdown("### Example questions")
    examples = [
        "What does this project do?",
        "What are the main files?",
        "How does authentication work?",
        "Where is the database code?",
        "How do I add a new feature?",
        "What dependencies does it use?",
    ]
    col1, col2 = st.columns(2)
    for i, q in enumerate(examples):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"💬 *{q}*")

else:
    st.subheader(f"💬 Ask about `{st.session_state.repo_name}`")

    # Quick question buttons
    st.markdown("**Quick questions:**")
    quick = [
        "What does this project do?",
        "What are the main files?",
        "What dependencies does it use?",
        "How is the code structured?",
    ]
    cols = st.columns(4)
    for i, q in enumerate(quick):
        with cols[i]:
            if st.button(q, use_container_width=True, key=f"quick{i}"):
                st.session_state.messages.append({
                    "role": "user", "content": q
                })
                with st.spinner("Reading code..."):
                    response = ask_question(
                        q,
                        st.session_state.vectorstore,
                        st.session_state.history
                    )
                st.session_state.messages.append({
                    "role": "assistant", "content": response
                })
                st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask anything about the code..."):
        st.session_state.messages.append({
            "role": "user", "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Reading code..."):
                response = ask_question(
                    question,
                    st.session_state.vectorstore,
                    st.session_state.history
                )
            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant", "content": response
        })
        st.rerun()