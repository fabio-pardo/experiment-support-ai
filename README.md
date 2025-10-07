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
a [uv](https://github.com/astral-sh/uv) installed.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd support-ai
    ```

2.  **Install dependencies using `uv`:**
    `uv` will automatically create a virtual environment (`.venv`) in the project directory and install the necessary dependencies.
    ```sh
    uv pip install -e '.[dev]'
    ```
    This command installs the project in editable mode (`-e`) along with the development dependencies.

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

`uv` automatically detects and uses the project's virtual environment, so you don't need to manually activate it.

By default, it will run with the sample ticket:
```sh
uv run rag-cli
```

You can also specify the path to a different support ticket using the `--ticket-path` argument:
```sh
uv run rag-cli --ticket-path /path/to/your/ticket.pdf
```
This will start the ingestion process and then run a query with the content of the specified ticket.

### Code Quality

This project uses `black` for code formatting and `ruff` for linting. Use `uv run` to execute them within the project's environment.

- **To format the code**, run the following command from the root of the project:
  ```sh
  uv run black .
  ```

- **To lint the code**, run:
  ```sh
  uv run ruff check .
  ```
