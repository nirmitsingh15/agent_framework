# AI Agent for Question Answering from PDF

### Problem Statement

The goal of this project is to create an AI agent that leverages the capabilities of a large language model (LLM) to extract answers based on the content of a large PDF document. The solution should use OpenAI's LLMs, and the logic for question answering should be implemented manually without using pre-built chains or frameworks such as Langchain or Llama Index. The code should be production-grade, clean, and modular.

### Solution Overview

This project implements an AI agent capable of answering a list of questions based on the content of a provided PDF document. The solution uses OpenAI's GPT-4o-mini model to process the content and generate answers. The agent performs the following tasks:

1. **Extracts the text** from the PDF document.
2. **Processes the list of questions** provided by the user.
3. **Splits the document** into chunks.
4. **Searches for relevant chunks** to answer the questions.
5. **Generates answers** for each question using OpenAI's GPT-4o-mini model.

The output is a structured response that pairs each question with its corresponding answer.

### Input Requirements

- **PDF file** containing the document over which the questions will be answered.
- **Text file** containing a list of questions.

### Supported Input File Types

- **PDF**: The input document is expected to be in PDF format.
- **Text File**: The list of questions should be provided in a plain text file.

### Ideal Output Format

The output will return a structured response where each question is paired with its corresponding answer. The agent will:

- Provide answers based on exact matches in the document.
- If an answer is not found or has low confidence, it will return **"Data Not Available"**.

### Instructions for Use

To run the agent, follow these steps:

1. Clone the repository:

   `git clone https://github.com/nirmitsingh15/agent_framework.git`


2. Install dependencies:

   The project requires OpenAI, PyPDF2, and other necessary libraries. Install them using:

   `pip install -r requirements.txt`

3. Run the program:

   To run the agent, use the following command:

   `python main.py --pdf_path <path_to_pdf_file> --questions_file <path_to_questions_file> --context <Any additional context you want to provide>`

   Replace `<path_to_pdf_file>` with the path to your PDF document and `<path_to_questions_file>` with the path to your text file containing the questions.

### Key Features

- **Recursive Text Chunking**: The document is divided into chunks for better context retrieval.
- **Word-to-Word Matching**: The model is instructed to prioritize exact matches from the context when generating answers.
- **Fallback for Low Confidence**: The model returns "Data Not Available" if the answer is not found with high confidence.

### Code Overview

The code is structured in multiple modules for clarity and separation of concerns:

- `main.py`: Contains the main logic to handle PDF processing, question answering, and calling the AI model.
- `pdf_processor.py`: Handles the reading and processing of the PDF document.
- `query_engine.py`: Responsible for querying the OpenAI API and retrieving answers.
- `chunker.py`: Contains the logic for chunking the PDF content into meaningful sections using a recursive text splitter
- `requirements.txt`: Installs necessary libraries
- `README.md`: Project documentation.

### Core Logic: How It Works

- **Text Extraction**: The `PDFReader` class is used to extract the content from the provided PDF. It handles text extraction page by page to ensure the entire document is processed.
- **Chunking**: The text is split into semantic chunks using a recursive method that ensures each chunk contains contextually related sentences. The chunking logic ensures that no chunk exceeds a defined token limit for processing by the GPT model.
- **Query Handling**: The `QueryEngine` class handles the user queries. It performs the following steps:
  - Receives a question.
  - Searches for the most relevant chunks of text from the PDF.
  - If an answer is found, it prioritizes word-to-word matches from the context.
  - If no relevant answer is found, it responds with "Data Not Available".
- **GPT Model Integration**: The solution leverages OpenAI's GPT-4o-mini model. The model processes the query and context to generate an answer. The output is a clean, structured response.

### Example Questions

- What is the name of the company?
- Who is the CEO of the company?
- What is the vacation policy?
- What is the termination policy?

### Project Structure
.
├── main.py                   # Main program logic to handle PDF processing and question answering
├── pdf_processor.py          # Handles PDF reading and processing
├── query_engine.py           # Module for querying the OpenAI API and retrieving the answers
├── chunker.py      # Contains logic for semantic chunking of the PDF content
├── README.md                 # Project documentation
└── input_files/             
    ├── handbook.pdf          # Input PDF document (handbook)
    └── questions.txt         # Sample text file with questions
└── output_files/             
    ├── answers.json         # Question-Answer pair
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies


### OpenAI API Key

Import your openapi key by following command

Example:

`export OPENAI_API_KEY="your-api-key-here"`

### Enhancements and Improvements

In a production-grade system, the following improvements could be made:

- **Modularization**: Further breaking down the components into microservices or separate modules for more scalable architecture.
- **Scalability**: Implementing support for processing multiple PDFs or large PDFs or other type of text documents
- **Accuracy Improvements**: Implementing a more advanced semantic search algorithm or fine-tuning the model for domain-specific tasks to improve accuracy.
- **Accuracy Improvements**: The initial approach of semantic chunking with clusters did not provide satisfactory results, and we can explore alternative semantic chunking methodologies. These include:
  - Using other semantic chunking methodologies
  - Leveraging other vector databases like Chroma, Pinecone, or Weaviate for better similarity searches and retrieval.
  - Improving the prompts to make them more specific and targeted for better question answering performance.
- **Logging and Monitoring**: Adding logging for better traceability and debugging, and monitoring for performance metrics.

