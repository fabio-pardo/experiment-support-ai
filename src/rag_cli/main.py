import argparse
import os
from pathlib import Path

import chromadb
import google.generativeai as genai
from chromadb.api.types import QueryResult
from dotenv import load_dotenv

from support_ai.config import COLLECTION_NAME, EMBEDDING_FN
from support_ai.ingest import process_ticket_pdf, run_document_ingestion

_ = load_dotenv()  # Load environment variables from .env

gemini_api_key = os.environ.get("GOOGLE_API_KEY")
if not gemini_api_key:
    print("Error: GOOGLE_API_KEY environment variable not set for Gemini.")

# Configure Gemini only once per run, or ensure it's configured.
# For this demonstration, we'll re-configure inside the function,
# but in a production app, you might configure it globally or pass the client.
try:
    genai.configure(  # pyright: ignore[reportPrivateImportUsage, reportUnknownMemberType]
        api_key=gemini_api_key
    )
except Exception as e:
    print("Error configuring Gemini API:", e)


def generate_llm_response(
    ticket_content: str,
    retrieved_results: list[tuple[str, dict[str, str], str]],
) -> str:
    """
    Generates a simulated LLM response based on the ticket content and retrieved information.
    In a real system, this would involve calling an actual LLM API.
    """

    system_prompt = """
    You are an expert IT support agent. Your task is to analyze a support ticket and provide the best course of action
    based on the provided context. The course of action should be concise, actionable, and refer to the context
    """

    user_query = f"Support Ticket:\n```\n{ticket_content}\n```\n\n"
    user_query += "Relevant Information from Knowledge Base:\n"

    if retrieved_results:
        for doc, meta, doc_id in retrieved_results:
            user_query += f"\n--- Context from {doc_id} (Type: {meta.get('type')} ---\n"
            user_query += f"{doc}\n"
        user_query += "\nBased on the above support ticket and the relevant information, what is the best course of action?"
    else:
        user_query += "No relevant information was found in the knowledge base.\n"
        user_query += "\nBased on the above support ticket, what is a general best course of action without specific context?"

    try:
        model = genai.GenerativeModel(  # pyright: ignore[reportPrivateImportUsage]
            "gemini-2.5-pro"
        )  # Using gemini-pro for text generation
        response = model.generate_content(  # pyright: ignore[reportUnknownMemberType]
            [system_prompt, user_query]
        )
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"


def run_workflow(ticket_path: Path):
    # Ingest documents into the vector database
    run_document_ingestion()
    print("Documents ingested successfully.")
    # Assuming you have a sample ticket PDF, e.g., in your data/pdfs directory
    ticket_content = process_ticket_pdf(ticket_path)
    if not ticket_content.startswith("Error"):
        print(
            f"\n--- Extracted Ticket Content from {ticket_path.name} (first 500 chars) ---"
        )
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
            name=COLLECTION_NAME,
            embedding_function=EMBEDDING_FN,  # pyright: ignore[reportArgumentType]
        )

        results_for_llm = []
        if not ticket_content.startswith("Error"):
            # Use the extracted ticket content as the query
            print("\n--- Querying with Ticket Content ---")
            print(f"Query: {ticket_content[:100]}...")  # Print a snippet of the query
            res: QueryResult = collection.query(
                query_texts=[ticket_content], n_results=5
            )  # Query with ticket content
            print("\nTop matches for the ticket:")

            if res["documents"] and res["metadatas"] and res["ids"]:
                results_for_llm = list(
                    zip(
                        res["documents"][0],
                        res["metadatas"][0],
                        res["ids"][0],
                        strict=False,
                    )
                )
                for doc, meta, id_ in results_for_llm:
                    print(f"- {id_}  ({meta.get('type') if meta else 'N/A'})")
                    print(
                        doc[:200].replace("\n", " ") + ("…" if len(doc) > 200 else "")
                    )
                    print()

        # Generate LLM response
        llm_output = generate_llm_response(
            ticket_content,
            # We are confident that the types are correct here
            results_for_llm,  # pyright: ignore[reportArgumentType]
        )
        print("\n--- LLM Generated Best Course of Action ---\n")
        print(llm_output)
    except Exception as e:
        print("Skipping query due to error in reading ticket content.")
        print(f"\n--- An error occurred during ChromaDB interaction: {e} ---")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Support AI Agent using RAG.")
    _ = parser.add_argument(
        "--ticket-path",
        type=Path,
        default=Path("data/pdfs/CMDR.pdf"),
        help="Path to the support ticket PDF file.",
    )
    args = parser.parse_args()

    run_workflow(args.ticket_path)  # pyright: ignore[reportAny]


if __name__ == "__main__":
    main()
