# Support AI Agent with Multimedia Support

## Overview

This project is a Support AI Agent that uses a Retrieval Augmented Generation (RAG) architecture. It ingests documents from a local `data/` directory, stores them in a vector database, and uses a Large Language Model (LLM) to answer queries based on the ingested content.

## Current Functionality

- **Data Ingestion**: The agent ingests various document types including PDFs, video transcripts (.vtt, .srt), text files (.txt, .md), and code files. It processes these files, chunks them, and stores them in a ChromaDB vector store.
- **RAG Pipeline**: When a query is provided (simulated via a sample support ticket PDF), the application searches the vector store for relevant documents.
- **LLM Integration**: The retrieved documents are then passed to a Large Language Model (Google's Gemini) along with the original query to generate a comprehensive answer.

## Technical Details

- **Vector Database**: Uses ChromaDB for local vector storage.
- **Embeddings**: Utilizes Sentence-Transformers (`all-MiniLM-L6-v2`) for generating document embeddings.
- **LLM**: Integrated with the Google Gemini API to generate responses.
- **Core Scripts**:
  - `ingest.py`: Handles the data ingestion pipeline.
  - `main.py`: Contains the main application logic for the RAG pipeline.
  - `config.py`: Manages configuration for the application.

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

- Python 3.12 or higher.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd support-ai
    ```

2.  **Create a virtual environment:**
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    The project uses the dependencies listed in `pyproject.toml`. You can install them using `pip`:
    ```sh
    pip install .
    ```

### Configuration

The application requires a Google Gemini API key.

1.  Copy the example `.env.example` file to `.env`:
    ```sh
    cp .env.example .env
    ```
2.  Open the `.env` file and add your Google Gemini API key:
    ```
    GOOGLE_API_KEY='your-api-key-here'
    ```

### Running the Application

Once the dependencies are installed and the `.env` file is configured, you can run the application from the command line.

By default, it will run with the sample ticket:
```sh
python main.py
```

You can also specify the path to a different support ticket using the `--ticket-path` argument:
```sh
python main.py --ticket-path /path/to/your/ticket.pdf
```
This will start the ingestion process and then run a query with the content of the specified ticket.
