# HTML-Based RAG Pipeline for Test Automation

An end-to-end Retrieval-Augmented Generation (RAG) pipeline designed to ingest offline HTML documentation, parse complex software specifications, and leverage local open-weight Large Language Models (LLMs) to automatically generate executable test run commands.

This project eliminates the manual overhead of mapping documentation requirements to CLI test suites, ensuring precise, context-aware command generation without relying on expensive, third-party cloud APIs.

## 🚀 Key Features

- **Local-First Architecture:** Keeps data 100% private by executing reasoning entirely on local or open-weight models (e.g., Llama 3, Mistral, Phi-3).
- **HTML Document Ingestion:** Specialized parsers to extract semantic meaning, code snippets, and structural metadata from local HTML documentation pages.
- **Context-Aware Retrieval:** Employs advanced text chunking and vector embeddings to pull exactly the relevant documentation sections needed for a given testing scenario.
- **CLI Command Generation:** Directly translates natural language test goals or requirement specs into structured, syntax-accurate test execution commands.

## 🏗️ System Architecture

1. **Document Ingestion:** Parses raw HTML files, cleans out boilerplate tags, and preserves hierarchical layouts (headings, tables, parameters).
2. **Chunking & Embedding:** Breaks documents into optimized segments and converts them into dense vector representations.
3. **Vector Database:** Indexing and storing embeddings for ultra-fast semantic searching.
4. **Retrieval & Synthesis:** Queries the vector index for target contexts based on the user's testing prompt.
5. **Local Inference Engine:** Passes the synthesized context to a local LLM via an inference framework (like Ollama, Hugging Face Transformers, or llama.cpp) to output the finalized CLI test command.

## 🛠️ Tech Stack

- **Language:** Python
- **Orchestration / RAG Framework:** LangChain
- **HTML Parsing:** BeautifulSoup4
- **Vector Storage:** ChromaDB
- **Local Inference Framework:** Ollama
- **Embeddings & LLMs:** BGE-Embeddings / Llama 3 (or specified open-weight model)

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/shindeavi47/RAG_Pipeline_HTML/
   cd RAG_Pipeline_HTML
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Local LLM:**
   Ensure you have your local inference environment configured. For example, if using Ollama:
   ```bash
   ollama run <local llm model name with version>
   ```

## 🔧 How to Run

**Generate Test Commands:** Execute the main pipeline script by querying the system with the test logic you need generated:
   ```bash
   python answer_generation.py  <prompt > (e.g. "Write a command to run tests in parallel with 2 threads")
   ```

## 🎯 Future Roadmap

- [ ] Support for dynamic HTML pages utilizing Javascript rendering (Playwright integration).
- [ ] Evaluation suite to score retrieval accuracy and command syntax correctness.
- [ ] UI Dashboard for browsing ingested documentation chunks and live testing.
