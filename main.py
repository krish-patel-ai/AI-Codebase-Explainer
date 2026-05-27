import os
import shutil
import time
import git
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# ── Models ────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=500)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Models ready!")

# ── Core functions ────────────────────────────────────────
def clone_repo(github_url):
    """Clone a GitHub repo to local folder"""
    repo_name  = github_url.rstrip("/").split("/")[-1]
    clone_path = f"cloned_repos/{repo_name}"
    if os.path.exists(clone_path):
        shutil.rmtree(clone_path)
    os.makedirs("cloned_repos", exist_ok=True)
    print(f"Cloning {repo_name}...")
    git.Repo.clone_from(github_url, clone_path)
    print(f"Done! Saved to: {clone_path}")
    return clone_path, repo_name


def load_code_files(repo_path):
    """Load all code files from the cloned repo"""
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
            print(f"Loaded {len(docs)} .{ext} files")
        except Exception as e:
            print(f"Skipped .{ext}: {e}")
            continue
    print(f"\nTotal files loaded: {len(all_docs)}")
    return all_docs


def split_code(all_docs):
    """Split documents into chunks using language-aware splitters"""
    EXTENSION_TO_LANGUAGE = {
        "py":   Language.PYTHON,
        "js":   Language.JS,
        "ts":   Language.TS,
        "jsx":  Language.JS,
        "tsx":  Language.TS,
        "java": Language.JAVA,
        "cpp":  Language.CPP,
        "c":    Language.CPP,
        "go":   Language.GO,
        "rb":   Language.RUBY,
        "rs":   Language.RUST,
        "md":   Language.MARKDOWN,
    }

    all_chunks = []
    for doc in all_docs:
        ext      = doc.metadata.get("file_type", "").lower()
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

    print(f"Original files : {len(all_docs)}")
    print(f"After splitting: {len(all_chunks)} chunks")
    return all_chunks


def store_in_chromadb(chunks):
    """Store code chunks in ChromaDB (in-memory)"""
    print("Storing chunks in ChromaDB...")
    time.sleep(1)  # ensure any previous instance is released
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print(f"Stored {len(chunks)} chunks ✅")
    return vectorstore


def ask_question(question, vectorstore, history):
    """Ask any question about the codebase"""
    start_search = time.time()
    # Step 1: Retrieve relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
    )
    docs = retriever.invoke(question)
    search_latency_ms = (time.time() - start_search) * 1000
    print(f"🔍 Vector DB Query Latency: {search_latency_ms:.2f} ms")

    # Step 2: Format context with file names
    context = "\n\n".join([
        f"# File: {d.metadata['file_name']}\n{d.page_content}"
        for d in docs
    ])

    # Step 3: Build prompt
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

    # Step 4: Run chain
    parser   = StrOutputParser()
    chain    = prompt | llm | parser
    start_llm = time.time()
    response = chain.invoke({
        "context" : context,
        "history" : history.messages,
        "question": question
    })
    print(f"🤖 LLM Generation Time: {time.time() - start_llm:.2f} seconds")

    # Step 5: Save to memory
    history.add_user_message(question)
    history.add_ai_message(response)

    return response


def build_codebase_explainer(github_url):
    """Complete pipeline in one function"""
    print(f"Building explainer for: {github_url}\n")
    start_ingestion = time.time()
    clone_path, repo_name = clone_repo(github_url)
    all_docs              = load_code_files(clone_path)
    chunks                = split_code(all_docs)
    vectorstore           = store_in_chromadb(chunks)
    history               = ChatMessageHistory()
    elapsed_ingestion = time.time() - start_ingestion

    print("\n" + "═" * 50)
    print(f"✅ Ready! Indexed {len(all_docs)} files, {len(chunks)} chunks")
    print(f"⏱  Total Ingestion Time: {elapsed_ingestion:.2f} seconds")
    print(f"Repo: {repo_name}")
    print("═" * 50 + "\n")

    return vectorstore, history, repo_name


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    vectorstore, history, repo_name = build_codebase_explainer(
        "https://github.com/psf/requests"
    )

    questions = [
        "What does this project do?",
        "What are the core source code files and what does each do?",
        "What language is it written in?",
        "How do I install this?",
        "Are there any tests?",
    ]

    print(f"REPO: {repo_name}\n")

    for i, q in enumerate(questions):
        start    = time.time()
        response = ask_question(q, vectorstore, history)
        elapsed  = time.time() - start
        print(f"Q{i+1}: {q}")
        print(f"A  : {response}")
        print(f"⏱  : {elapsed:.2f}s\n")