# Business-Travel-Assistant
**Business Travel Assistant | RAG-Based GenAI Application**  A Generative AI-powered assistant that retrieves information from business travel policy documents and provides context-based answers. Built using Python, Flask, LangChain, LangGraph, ChromaDB and Ollama.  Explore the source code and project implementation.
# Business Travel Assistant

An AI-powered Business Travel Assistant that helps employees understand company travel policies and estimate hotel reimbursement.

## Features
- Policy-based question answering using RAG.
- Document retrieval using ChromaDB.
- Local LLM integration using Ollama.
- Hotel reimbursement estimation.
- Input validation and policy-based responses.
- Workflow orchestration using LangGraph.
- Web interface built using Flask.

## Technologies
- Python
- Flask
- LangChain
- LangGraph
- Ollama
- ChromaDB
- HTML and CSS

## Project Structure

```text
project04_travel_assistant/
├── app.py
├── graph.py
├── rag.py
├── tools.py
├── ingest.py
├── requirements.txt
├── data/
├── templates/
├── .env.example
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Ollama and pull the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

5. Generate the vector database:

```bash
python ingest.py
```

6. Start the Flask application:

```bash
python app.py
```

7. Open `http://127.0.0.1:5000` in your browser.

## Note

Ollama must be installed and running locally. The ChromaDB database is generated from the policy documents during ingestion.

This project is intended as an AI-assisted policy information and reimbursement estimation tool. It does not approve expenses or book travel.
