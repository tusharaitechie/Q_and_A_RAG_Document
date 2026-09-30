# 🤖 RAG Document Q&A System

### Retrieval-Augmented Generation (RAG) based Question Answering over Research Papers

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)](https://faiss.ai/)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036)](https://groq.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)

> **An end-to-end RAG application that allows users to ask natural-language questions about research papers and receive answers grounded in the retrieved document context.**

---

## 📌 Project Overview

Reading and searching through large research papers manually can be time-consuming. A traditional keyword search may also fail when the user's question is expressed differently from the wording used in the document.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**.

The application:

1. Loads PDF research papers from a local directory.
2. Splits the documents into smaller overlapping chunks.
3. Converts each chunk into a numerical vector using embeddings.
4. Stores the vectors in a **FAISS vector index**.
5. Converts the user's question into a semantic search query.
6. Retrieves the most relevant document chunks.
7. Sends the retrieved context to a **Groq-hosted LLM**.
8. Generates an answer using the retrieved context.
9. Displays the answer and the retrieved document passages in the Streamlit UI.

This approach helps the LLM answer questions using the information contained in the uploaded research material rather than relying only on its pretrained knowledge.

---

## 🎯 Key Objectives

- Build a practical **Retrieval-Augmented Generation** pipeline.
- Enable semantic question answering over research papers.
- Understand document ingestion and chunking.
- Generate vector embeddings for document chunks.
- Perform similarity-based retrieval using FAISS.
- Integrate an LLM through the Groq API.
- Build an interactive UI using Streamlit.
- Display the retrieved context for better transparency and debugging.

---

## ✨ Features

### 📄 Document Ingestion

Research papers stored inside the `research_papers/` directory are loaded using LangChain's `PyPDFDirectoryLoader`.

### ✂️ Intelligent Text Chunking

Documents are split using `RecursiveCharacterTextSplitter`.

Current configuration:

- **Chunk size:** 1000 characters
- **Chunk overlap:** 200 characters

The overlap helps preserve contextual information between neighboring chunks.

### 🧠 Semantic Embeddings

The project supports two embedding approaches:

#### Option 1 — OpenAI Embeddings

The main implementation uses:

```python
OpenAIEmbeddings()
```

This converts text chunks into numerical vectors that capture their semantic meaning.

#### Option 2 — Hugging Face Embeddings

The alternative implementation uses:

```python
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

This provides a practical embedding option using the `all-MiniLM-L6-v2` model.

### 🔎 Vector Similarity Search

The generated embeddings are stored using:

```python
FAISS.from_documents(...)
```

FAISS enables efficient similarity-based retrieval of relevant document chunks.

### 🤖 LLM-Powered Answer Generation

The retrieved context is passed to a Groq-powered chat model:

```python
ChatGroq(
    model_name="openai/gpt-oss-20b"
)
```

The prompt instructs the model to answer using the supplied context.

### 🖥️ Interactive Streamlit Interface

The application provides:

- A text input for questions
- A document embedding button
- Generated answers
- An expandable section containing retrieved document passages

### 🔍 Retrieved Context Visibility

The application exposes the retrieved context under:

> **Document similarity Search**

This is useful for understanding which passages were supplied to the LLM.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Research Papers    │
                    │       (PDFs)         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  PyPDFDirectoryLoader│
                    │   Document Loading   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  RecursiveCharacter  │
                    │    Text Splitter     │
                    │  Chunk: 1000 chars   │
                    │ Overlap: 200 chars   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Text Embeddings    │
                    │ OpenAI / HuggingFace │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │        FAISS         │
                    │   Vector Store /     │
                    │   Similarity Search  │
                    └──────────┬───────────┘
                               │
              User Question   │
                    ┌──────────▼───────────┐
                    │      Retriever       │
                    │ Relevant Chunks      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Retrieved Context  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Groq LLM         │
                    │   Answer Generation  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Streamlit Response  │
                    │ Answer + Context     │
                    └──────────────────────┘
```

---

# 🔄 RAG Pipeline Explained

## 1. Document Loading

PDF research papers are placed inside:

```text
research_papers/
```

The application loads them using:

```python
PyPDFDirectoryLoader("research_papers")
```

This is the ingestion stage of the RAG pipeline.

---

## 2. Text Splitting

Large documents are not directly passed to the LLM.

Instead, they are divided into smaller chunks:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

### Why chunking?

A research paper can contain thousands of words. Sending the entire document for every question would:

- Increase token usage
- Increase latency
- Make retrieval less precise
- Potentially exceed model context limits

Chunking makes relevant information easier to retrieve.

### Why overlap?

Suppose an important sentence starts near the end of one chunk and continues into the next.

Without overlap, some contextual information may be lost.

With a 200-character overlap, neighboring chunks share part of their content.

---

## 3. Embedding Generation

Each text chunk is converted into a vector representation.

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
[0.12, -0.43, 0.87, ...]
```

The vector represents the semantic meaning of the text.

This enables the system to retrieve conceptually similar content even when the wording is not identical.

---

## 4. Vector Store — FAISS

The generated embeddings are stored in a FAISS index:

```python
FAISS.from_documents(
    final_documents,
    embeddings
)
```

FAISS is used for efficient similarity search over vectors.

When a user asks a question, the system searches the vector index to find document chunks that are semantically related to the query.

---

## 5. Query Retrieval

The user's question is passed to the retriever:

```python
retriever = vectors.as_retriever()
```

The retriever identifies relevant document chunks.

Conceptually:

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant Document Chunks
```

---

## 6. Context Injection

The retrieved chunks are supplied to the document chain.

The prompt explicitly instructs the LLM:

```text
Answer the questions based on the provided context only.
```

This is an important RAG design principle because the retrieved documents become the information source for answer generation.

---

## 7. Answer Generation

The retrieved context and user question are passed to the Groq-powered LLM.

The generation flow is:

```text
Retrieved Context
       +
User Question
       ↓
    Groq LLM
       ↓
Generated Answer
```

---

# 🧩 Core RAG Components

| Component | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Interactive web interface |
| Document Loader | PyPDFDirectoryLoader | Load PDF research papers |
| Text Splitter | RecursiveCharacterTextSplitter | Split documents into chunks |
| Embeddings | OpenAI / Hugging Face | Convert text into vectors |
| Vector Store | FAISS | Similarity search |
| LLM | Groq | Generate answers |
| Orchestration | LangChain | Connect RAG components |
| Configuration | python-dotenv | Load API keys from `.env` |

---

# 🛠️ Tech Stack

### Programming Language
- Python

### Generative AI / LLM
- Groq API
- `ChatGroq`
- `openai/gpt-oss-20b`

### RAG / LLM Framework
- LangChain

### Embeddings
- OpenAI Embeddings
- Hugging Face `all-MiniLM-L6-v2`

### Vector Database / Search
- FAISS

### Document Processing
- PyPDFDirectoryLoader
- RecursiveCharacterTextSplitter

### Frontend
- Streamlit

### Environment Management
- python-dotenv

---

# 📂 Project Structure

```text
Q_and_A_RAG_Document/
│
├── 📁 research_papers/
│   └── Your PDF research papers
│
├── 📄 main.py
│   └── Main RAG application
│
├── 📄 app_huggingfaceembedding.py
│   └── RAG application using Hugging Face embeddings
│
├── 📄 requirements.txt
│   └── Python dependencies
│
├── 📄 .gitignore
│   └── Files excluded from Git
│
└── 📄 README.md
    └── Project documentation
```

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/tusharaitechie/Q_and_A_RAG_Document.git
```

Move into the project directory:

```bash
cd Q_and_A_RAG_Document
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
```

### Important

Never commit your actual API keys to GitHub.

Add `.env` to `.gitignore`:

```text
.env
```

---

# 📚 Add Research Papers

Place your PDF documents inside:

```text
research_papers/
```

Example:

```text
research_papers/
├── paper_01.pdf
├── paper_02.pdf
└── paper_03.pdf
```

The application loads the available PDF documents from this directory.

---

# ▶️ Run the Application

## Main Version — OpenAI Embeddings

Run:

```bash
streamlit run main.py
```

Streamlit will start the application locally.

Open the URL displayed in your terminal, typically:

```text
http://localhost:8501
```

---

## Alternative Version — Hugging Face Embeddings

If you want to use the local Hugging Face embedding model:

```bash
streamlit run app_huggingfaceembedding.py
```

This version uses:

```text
all-MiniLM-L6-v2
```

for generating embeddings.

---

# 🧪 How to Use

### Step 1 — Start the application

Run:

```bash
streamlit run main.py
```

### Step 2 — Build the vector index

Click:

> **Document Embedding**

The application will:

```text
Load PDFs
   ↓
Split Documents
   ↓
Generate Embeddings
   ↓
Create FAISS Index
```

### Step 3 — Ask a question

Enter a question related to the research papers.

For example:

```text
What is the main objective of the research?
```

or:

```text
What methodology was used in the study?
```

### Step 4 — Review the answer

The application generates an answer based on the retrieved context.

### Step 5 — Inspect retrieved context

Expand:

> **Document similarity Search**

to see the document passages retrieved for the query.

---

# 💡 Example RAG Flow

Suppose a research paper contains information about:

```text
Transformer-based architectures for NLP
```

The user asks:

```text
How do transformer models improve NLP tasks?
```

The system performs:

```text
Question
   ↓
Question Embedding
   ↓
FAISS Similarity Search
   ↓
Relevant Research Paper Chunks
   ↓
Context + Question
   ↓
Groq LLM
   ↓
Grounded Answer
```

The important point is that the LLM is provided with retrieved document context before generating the response.

---

# 🧠 Why RAG Instead of Direct LLM Question Answering?

A normal LLM workflow looks like:

```text
Question
   ↓
LLM
   ↓
Answer
```

The model mainly relies on its pretrained knowledge.

A RAG workflow looks like:

```text
Question
   ↓
Retrieve Relevant Information
   ↓
Add Retrieved Context
   ↓
LLM
   ↓
Answer
```

This makes RAG particularly useful for:

- Private documents
- Research papers
- Company knowledge bases
- Technical documentation
- Legal documents
- Internal reports
- Frequently updated information

---

# 🛡️ Hallucination Reduction

The project uses a grounding-oriented prompt:

```text
Answer the questions based on the provided context only.
```

The purpose is to encourage the model to generate answers from the retrieved document context instead of freely generating information unrelated to the source material.

> **Important:** Prompt-based grounding can reduce unsupported answers, but it does not guarantee zero hallucinations.

---

# ⚡ Performance Considerations

The project records processing time around the retrieval-and-generation operation:

```python
start = time.process_time()

response = retrieval_chain.invoke({
    "input": user_prompt
})

print(f"Response time :{time.process_time()-start}")
```

This makes it possible to inspect execution time while experimenting with:

- Different embedding models
- Chunk sizes
- Chunk overlap
- Retrieval settings
- LLM configurations

---

# 🔬 Engineering Concepts Demonstrated

This project demonstrates practical understanding of several GenAI and ML engineering concepts:

### RAG
Retrieval-Augmented Generation combines information retrieval with LLM generation.

### Chunking
Large documents are divided into manageable semantic units.

### Embeddings
Text is represented as numerical vectors for semantic comparison.

### Vector Search
FAISS finds document chunks that are similar to the user's query.

### Retrieval
Relevant chunks are selected before answer generation.

### Prompt Grounding
Retrieved context is explicitly supplied to the LLM.

### LLM Integration
The project integrates a Groq-hosted language model through LangChain.

### Session State
Streamlit session state is used to retain embeddings, loaded documents, and the FAISS vector store during interaction.

---

# 📈 Current Implementation

| Area | Current Implementation |
|---|---|
| Document Type | PDF |
| Document Source | Local `research_papers/` directory |
| Chunk Size | 1000 characters |
| Chunk Overlap | 200 characters |
| Vector Store | FAISS |
| Embeddings | OpenAI / Hugging Face |
| LLM | Groq `openai/gpt-oss-20b` |
| UI | Streamlit |
| Framework | LangChain |
| Retrieval | LangChain Retriever |
| Context Display | Yes |

---

# 🚀 Future Improvements

The current implementation can be extended into a more production-oriented RAG system.

### 🔹 1. Persistent Vector Store

Persist FAISS indexes so embeddings do not need to be regenerated after every application restart.

### 🔹 2. More Document Formats

Add support for:

- DOCX
- TXT
- CSV
- Markdown
- Web pages

### 🔹 3. Metadata Filtering

Store metadata such as:

```text
Document name
Page number
Section
Author
Publication year
```

and use it during retrieval.

### 🔹 4. Hybrid Search

Combine:

```text
Semantic Search
+
Keyword Search
```

to improve retrieval for technical terms and exact phrases.

### 🔹 5. Re-ranking

Retrieve a larger candidate set and use a re-ranker to select the most relevant chunks.

```text
FAISS Retrieval
      ↓
Top-N Candidates
      ↓
Re-ranker
      ↓
Top-K Context
      ↓
LLM
```

### 🔹 6. Source Citations

Display:

```text
Answer
+
Document Name
+
Page Number
+
Retrieved Passage
```

for stronger traceability.

### 🔹 7. Conversation Memory

Support follow-up questions such as:

```text
User: What methodology was used?

User: Why was this methodology selected?

User: What were its limitations?
```

### 🔹 8. Evaluation

Add RAG evaluation metrics such as:

- Context relevance
- Context recall
- Faithfulness
- Answer relevance
- Retrieval accuracy

---

# 🎤 Interview Explanation

### Tell me about this project.

> **“I built a Retrieval-Augmented Generation based document question-answering system for research papers. The application loads PDF documents, splits them into overlapping chunks, generates embeddings, and stores those embeddings in a FAISS vector index. When a user asks a question, the system retrieves semantically relevant document chunks and passes them along with the question to a Groq-hosted LLM through LangChain. The model then generates an answer grounded in the retrieved context. I also added a Streamlit interface that allows users to build the vector index, ask questions, and inspect the retrieved document context.”**

---

## ❓ Why did you use RAG?

> **“RAG allows the LLM to retrieve relevant information from an external knowledge source before generating an answer. This is useful when the required information is contained in private or domain-specific documents that may not be part of the model's pretrained knowledge.”**

---

## ❓ Why did you use FAISS?

> **“FAISS is designed for efficient similarity search over high-dimensional vectors. In this project, I use it to retrieve document chunks whose embeddings are semantically similar to the user's question.”**

---

## ❓ Why do you chunk documents?

> **“Large documents are difficult and inefficient to send directly to an LLM. Chunking divides them into smaller units so that the retrieval system can identify the most relevant pieces of information for a particular question.”**

---

## ❓ Why use chunk overlap?

> **“Overlap helps preserve context across chunk boundaries. If an important sentence or concept spans two chunks, overlap reduces the probability of losing that contextual relationship.”**

---

## ❓ Why use embeddings?

> **“Embeddings convert text into numerical vector representations that capture semantic relationships. This allows the system to retrieve text based on meaning rather than only exact keyword matches.”**

---

## ❓ What happens when a user asks a question?

> **“The question is converted into a representation suitable for retrieval, the vector store retrieves relevant document chunks, and those chunks are passed as context along with the question to the LLM. The LLM then generates the final response based on that retrieved context.”**

---

# 🏆 Skills Demonstrated

This project demonstrates hands-on experience with:

- 🐍 Python
- 🤖 Generative AI
- 🧠 Large Language Models
- 🔎 Retrieval-Augmented Generation
- 📚 LangChain
- 🧮 Vector Embeddings
- ⚡ FAISS
- 🦙 Hugging Face Embeddings
- 🚀 Groq API
- 📄 PDF Document Processing
- ✂️ Text Chunking
- 🔍 Semantic Search
- 🖥️ Streamlit
- 🔐 Environment Variables
- 🧩 AI Application Development

---

# 🔗 Repository

**GitHub:**  
https://github.com/tusharaitechie/Q_and_A_RAG_Document

---

# 👨‍💻 Author

**Tushar Nile**

Software Engineer | AI/ML & Generative AI Enthusiast

---

## ⭐ If you find this project useful

Feel free to explore the repository, experiment with different embedding models, chunking strategies, retrieval configurations, and LLMs.

---

### 📌 Project Summary

```text
PDF Research Papers
        ↓
Document Loading
        ↓
Text Chunking
        ↓
Embeddings
        ↓
FAISS Vector Store
        ↓
Similarity Retrieval
        ↓
Relevant Context
        ↓
Groq LLM
        ↓
Grounded Answer
        ↓
Streamlit UI
```

**This project demonstrates the complete fundamental RAG workflow from document ingestion to retrieval and LLM-based answer generation.**
