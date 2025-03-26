"""
Main script for running the Knowledge Graph-based Retrieval-Augmented Generation (RAG) system as a FastAPI service.
This service processes a PDF document, builds a knowledge base, and answers audit-related questions
using OpenAI's GPT models and a semantic knowledge graph.

Endpoints:
    - /process_document: Process a PDF document and build a knowledge base.
    - /run_analysis: Run analysis on questions and return results.
    - /get_statistics: Get summary statistics of the analysis.

Usage:
    Run the FastAPI server:
        uvicorn app.main:app --reload
"""

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from .audit_analysis_runner import AuditAnalysisRunner
import os
import shutil

app = FastAPI()

# Global variables for the runner
runner = None

@app.post("/process_document")
async def process_document(pdf_file: UploadFile, questions_file: UploadFile):
    """
    Endpoint to process a PDF document and build a knowledge base.

    Args:
        pdf_file (UploadFile): The PDF document to process.
        questions_file (UploadFile): The CSV file containing questions.

    Returns:
        JSONResponse: A message indicating the document has been processed.
    """
    global runner

    # Save uploaded files locally
    pdf_path = f"temp_{pdf_file.filename}"
    questions_path = f"temp_{questions_file.filename}"
    with open(pdf_path, "wb") as pdf_out:
        shutil.copyfileobj(pdf_file.file, pdf_out)
    with open(questions_path, "wb") as questions_out:
        shutil.copyfileobj(questions_file.file, questions_out)

    # Initialize the runner
    runner = AuditAnalysisRunner(pdf_path, questions_path)
    runner.setup()
    runner.process_document()

    return JSONResponse({"message": "Document processed and knowledge base built successfully."})


@app.post("/run_analysis")
async def run_analysis(output_path: str = Form("audit_results_openai.csv")):
    """
    Endpoint to run analysis on the questions and save results.

    Args:
        output_path (str): Path to save the output CSV file.

    Returns:
        JSONResponse: A message indicating the analysis has been completed.
    """
    global runner
    if not runner:
        return JSONResponse({"error": "No document has been processed yet."}, status_code=400)

    runner.output_path = output_path
    runner.run_analysis()

    return JSONResponse({"message": f"Analysis completed. Results saved to {output_path}."})


@app.get("/get_statistics")
async def get_statistics():
    """
    Endpoint to get summary statistics of the analysis.

    Returns:
        JSONResponse: Summary statistics of the analysis.
    """
    global runner
    if not runner:
        return JSONResponse({"error": "No analysis has been run yet."}, status_code=400)

    stats = runner.get_summary_statistics()
    return JSONResponse(stats)