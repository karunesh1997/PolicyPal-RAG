# 🏢 PolicyPal

### AI-Powered Enterprise Policy Assistant using RAG

PolicyPal is a Retrieval-Augmented Generation (RAG) application that allows employees to ask questions about company policies and receive source-grounded answers from internal policy documents.

---

## 🎯 Problem

Employees often spend significant time searching through company documents to find answers about:

- Leave policies
- Work-from-home policies
- Travel policies
- Expense policies
- Employee handbook
- HR policies
- Company procedures

PolicyPal provides a conversational interface for retrieving this information.

---

## 💡 Solution

PolicyPal uses Retrieval-Augmented Generation to:

1. Read company policy PDFs
2. Extract text and page metadata
3. Split documents into chunks
4. Generate vector embeddings
5. Store embeddings in FAISS
6. Retrieve relevant policy sections
7. Pass retrieved context to a local LLM
8. Generate a grounded answer
9. Display source documents and page numbers

---

## 🏗️ Architecture

```text
PDF Documents
      |
      v
PDF Text Extraction
      |
      v
Chunking + Metadata
      |
      v
Sentence Transformers
      |
      v
FAISS Vector Database
      |
      v
Semantic Retrieval
      |
      v
Relevant Policy Context
      |
      v
Ollama Local LLM
      |
      v
Answer + Sources
      |
      v
Streamlit UI