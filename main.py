import warnings
warnings.filterwarnings("ignore") 
import argparse
import json
from pdf_processor import PDFProcessor
from chunker import RecursiveTextChunker
from vector_store import VectorStore
from query_engine import QueryEngine

def read_questions(file_path):
    """Reads questions line-by-line from a text file."""
    with open(file_path, "r") as file:
        questions = [line.strip() for line in file if line.strip()]
    return questions

def exact_match_search(text_chunks, question):
    """
    Checks if the question has an exact word-for-word match in the text chunks.
    Returns the exact match if found.
    """
    for chunk in text_chunks:
        if question in chunk:
            return chunk
    return None

def main():
    parser = argparse.ArgumentParser(description="AI Agent for PDF Question Answering with Context")
    parser.add_argument("--pdf_path", required=True, help="Path to the PDF file")
    parser.add_argument("--questions_file", required=True, help="Path to the text file containing questions")
    parser.add_argument("--user_instructions", required=False, help="Additional context or instructions for the query")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    questions_file = args.questions_file
    user_instructions = args.user_instructions

    # Step 1: Extract text from PDF
    print("Extracting text from PDF...")
    pdf_processor = PDFProcessor()
    raw_text = pdf_processor.extract_text(pdf_path)

    # Step 2: Split text into chunks
    print("Splitting text into chunks...")
    chunker = RecursiveTextChunker()
    chunks = chunker.split_text(raw_text)

    # Step 3: Build Vector Store
    print("Building vector store...")
    vector_store = VectorStore()
    vector_store.add_texts(chunks)

    # Step 4: Read questions from file
    print(f"Reading questions from: {questions_file}")
    questions = read_questions(questions_file)

    # Step 5: Query Engine Initialization
    query_engine = QueryEngine(vector_store)

    # Step 6: Generate structured JSON output
    responses = {}
    for i, question in enumerate(questions, start=1):
        print(f"Processing Question {i}: {question}")
        
        # Check for exact word-for-word match first
        exact_match = exact_match_search(chunks, question)
        if exact_match:
            responses[question] = exact_match  # Word-for-word match found
            continue

        # If no exact match, query the engine with context
        full_query = f"{user_instructions}. {question}"
        answer = query_engine.answer_question(full_query)

        # Handle low-confidence responses
        if not answer or "Data Not Available" in answer:
            responses[question] = "Data Not Available"
        else:
            responses[question] = answer

    # Step 7: Output structured JSON
    output_file = "output_files/answers.json"
    with open(output_file, "w") as json_file:
        json.dump(responses, json_file, indent=4)

    print(f"\nStructured answers saved to {output_file}.")
    print(json.dumps(responses, indent=4))  # Print to console as well

if __name__ == "__main__":
    main()
