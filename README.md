---
title: Codebase Explainer
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🔍 AI Codebase Explainer

> Ask questions about any GitHub repository in plain English. No more reading 50 files manually.

## 🚀 Live Demo
👉 [Try it here](https://huggingface.co/spaces/Krishp1/codebase-explainer)

## 📊 Benchmark Results (psf/requests)
| Metric | Result |
|--------|--------|
| Files indexed | 46 files |
| Chunks created | 341 chunks |
| Ingestion time | 15.45 seconds |
| Vector DB latency | ~100ms |
| Accuracy | 4/5 questions ✅ |

## 🛠️ Tech Stack
- LangChain + RAG pipeline
- ChromaDB (vector store)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq LLaMA 3 (LLM)
- Streamlit (UI)
- Docker + HuggingFace Spaces (deployment)

## 💡 How It Works
1. Paste any GitHub repo URL
2. App clones and indexes all code files
3. Ask questions in plain English
4. Get answers with source file citations

## 🎯 Use Cases
- Understand a new codebase in minutes
- Onboard to a new company faster
- Explore open source projects instantly

## 🔗 Connect
👤 [LinkedIn](https://www.linkedin.com/in/krish-patel-4951713b3/)
