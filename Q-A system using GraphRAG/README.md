# Knowledge Graph RAG System

This project implements a Question-Answering (Q&A) system using a Knowledge Graph and OpenAI's GPT models. It processes documents, builds a knowledge base, and answers audit-related questions.

## Features

- Extracts text from PDF documents.
- Builds a semantic knowledge graph.
- Uses OpenAI embeddings for similarity search.
- Provides structured audit analysis in JSON format.
- Exposes a FastAPI-based web service for document processing and analysis.

## Endpoints

The FastAPI service provides the following endpoints:

1. **`POST /process_document`**  
   Upload a PDF document and a CSV file of questions to process and build a knowledge base.

2. **`POST /run_analysis`**  
   Run analysis on the uploaded questions and save the results to a CSV file.

3. **`GET /get_statistics`**  
   Retrieve summary statistics of the analysis.

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (add it to the `.env` file)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Q-A system using GraphRAG
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```properties
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the FastAPI Server

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Access the API documentation at:
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Usage

### Example Workflow

1. **Process a Document**  
   Use the `/process_document` endpoint to upload a PDF document and a CSV file of questions.

2. **Run Analysis**  
   Use the `/run_analysis` endpoint to analyze the questions and save the results.

3. **Get Statistics**  
   Use the `/get_statistics` endpoint to retrieve summary statistics of the analysis.

### Example cURL Commands

1. **Process a Document**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/process_document" \
        -F "pdf_file=@path_to_pdf.pdf" \
        -F "questions_file=@path_to_questions.csv"
   ```

2. **Run Analysis**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/run_analysis" \
        -F "output_path=audit_results.csv"
   ```

3. **Get Statistics**:
   ```bash
   curl -X GET "http://127.0.0.1:8000/get_statistics"
   ```

## File Structure

```
Q-A system using GraphRAG/
├── app/
│   ├── main.py                # FastAPI application
│   ├── config.py              # Configuration and constants
│   ├── utils.py               # Utility functions
│   ├── document_processor.py  # PDF processing logic
│   ├── semantic_chunking.py   # Text chunking logic
│   ├── knowledge_graph.py     # Knowledge graph implementation
│   ├── enhanced_audit_analyzer.py  # Core analysis logic
│   ├── audit_analysis_runner.py    # Orchestrates the analysis process
├── .env                       # Environment variables (e.g., OpenAI API key)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Dependencies

The project uses the following Python libraries:

- **PyPDF2**: For PDF processing.
- **pandas**: For data manipulation.
- **openai**: For OpenAI API integration.
- **networkx**: For graph operations.
- **chromadb**: For vector database operations.
- **langchain**: For document processing and embeddings.
- **scikit-learn**: For machine learning utilities.
- **nltk**: For natural language processing.
- **sentence-transformers**: For text embeddings.
- **tqdm**: For progress bars.
- **python-dotenv**: For managing environment variables.
- **matplotlib**: For graph visualization.
- **seaborn**: For advanced plotting.
- **fastapi**: For building the API.
- **uvicorn**: For running the FastAPI server.

## License

This project is licensed under the MIT License.