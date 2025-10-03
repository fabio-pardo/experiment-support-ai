from dotenv import load_dotenv
import chromadb
from pathlib import Path
from ingest import run_document_ingestion, process_ticket_pdf
from config import COLLECTION_NAME, embedding_fn

import os
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env


def generate_llm_response(
    ticket_content: str, retrieved_results: list[tuple[str, dict, str]]
) -> str:
    """
    Generates a simulated LLM response based on the ticket content and retrieved information.
    In a real system, this would involve calling an actual LLM API.
    """
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        return "Error: GOOGLE_API_KEY environment variable not set for Gemini."

    # Configure Gemini only once per run, or ensure it's configured.
    # For this demonstration, we'll re-configure inside the function,
    # but in a production app, you might configure it globally or pass the client.
    genai.configure(api_key=gemini_api_key)

    system_prompt = """
    You are an expert IT support agent. Your task is to analyze a support ticket and provide the best course of action
    based on the provided context. The course of action should be concise, actionable, and refer to the context
    """

    user_query = f"Support Ticket:\n```\n{ticket_content}\n```\n\n"
    user_query += "Relevant Information from Knowledge Base:\n"

    if retrieved_results:
        for i, (doc, meta, doc_id) in enumerate(retrieved_results):
            user_query += f"\n--- Context from {doc_id} (Type: {meta.get('type')} ---\n"
            user_query += f"{doc}\n"
        user_query += "\nBased on the above support ticket and the relevant information, what is the best course of action?"
    else:
        user_query += "No relevant information was found in the knowledge base.\n"
        user_query += "\nBased on the above support ticket, what is a general best course of action without specific context?"

    try:
        model = genai.GenerativeModel(
            "gemini-2.5-pro"
        )  # Using gemini-pro for text generation
        response = model.generate_content([system_prompt, user_query])
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"


def main():
    # Ingest documents into the vector database
    run_document_ingestion()
    print("Documents ingested successfully.")
    # Assuming you have a sample ticket PDF, e.g., in your data/pdfs directory
    sample_ticket_path = Path(
        "data/pdfs/CMDR.pdf"
    )  # Using an existing PDF as an example
    ticket_content = process_ticket_pdf(sample_ticket_path)
    if not ticket_content.startswith("Error"):
        print("\n--- Extracted Ticket Content (first 500 chars) ---")
        print(
            ticket_content[:500] + "..."
            if len(ticket_content) > 500
            else ticket_content
        )
        print("---------------------------------------------------")
    else:
        print(ticket_content)
    # --- End new code for processing a ticket ---

    try:
        # Query the vector database with the ticket content
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )

        if not ticket_content.startswith("Error"):
            # Use the extracted ticket content as the query
            print("\n--- Querying with Ticket Content ---")
            print(f"Query: {ticket_content[:100]}...")  # Print a snippet of the query
            res = collection.query(
                query_texts=[ticket_content], n_results=5
            )  # Query with ticket content
            print("\nTop matches for the ticket:")
            for doc, meta, id_ in zip(
                res["documents"][0], res["metadatas"][0], res["ids"][0]
            ):
                print(f"- {id_}  ({meta.get('type')})")
                print(doc[:200].replace("\n", " ") + ("…" if len(doc) > 200 else ""))
                print()

        # Generate LLM response
        llm_output = generate_llm_response(
            ticket_content,
            list(zip(res["documents"][0], res["metadatas"][0], res["ids"][0])),
        )
        print("\n--- LLM Generated Best Course of Action ---\n")
        print(llm_output)
    except Exception as e:
        print("Skipping query due to error in reading ticket content.")
        print(f"\n--- An error occurred during ChromaDB interaction: {e} ---")


if __name__ == "__main__":
    main()
