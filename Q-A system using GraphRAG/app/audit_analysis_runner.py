"""
Audit analysis runner module for the Knowledge Graph RAG system.
This module orchestrates the end-to-end process of document processing, knowledge base creation,
and answering audit-related questions.
"""

from typing import List, Dict
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from .enhanced_audit_analyzer import EnhancedAuditAnalyzer
from .config import OPENAI_API_KEY

class AuditAnalysisRunner:
    """
    The AuditAnalysisRunner class orchestrates the end-to-end process of document processing,
    knowledge base creation, and answering audit-related questions.
    """

    def __init__(self, pdf_path: str, questions_path: str, output_path: str = None):
        """
        Initializes the AuditAnalysisRunner with the paths to the PDF document, questions file,
        and an optional output path for the results.

        :param pdf_path: Path to the PDF document to be analyzed.
        :param questions_path: Path to the CSV file containing the questions.
        :param output_path: Optional path to save the analysis results. Defaults to a timestamped CSV file.
        """
        self.pdf_path = pdf_path
        self.questions_path = questions_path
        self.output_path = output_path or f"audit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.analyzer = None
        self.sentence_model = None

    def setup(self):
        """
        Sets up the necessary components for the analysis, including the sentence transformer model
        and the enhanced audit analyzer.
        """
        self.sentence_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
        self.analyzer = EnhancedAuditAnalyzer(embeddings=OPENAI_API_KEY)

    def load_questions(self) -> List[Dict]:
        """
        Loads the questions from the CSV file specified by the questions_path.

        :return: A list of dictionaries, each containing a question.
        """
        df = pd.read_csv(self.questions_path)
        return df.to_dict('records')

    def process_document(self):
        """
        Processes the PDF document to extract chunks of text and builds the knowledge base
        using the extracted chunks.
        """
        chunks = self.analyzer.process_document(self.pdf_path, self.sentence_model)
        self.analyzer.build_knowledge_base(chunks, self.pdf_path)

    def run_analysis(self):
        """
        Runs the analysis on the loaded questions, processes each question using the analyzer,
        and saves the results to the output CSV file.
        """
        questions = self.load_questions()
        results = []
        for question in tqdm(questions, desc="Analyzing questions"):
            result = self.analyzer.analyze_query(question['Question'])
            results.append({
                'Question': question['Question'],
                'Findings': result['findings'],
                'Conclusion': result['conclusion'],
                'Result': result['result'],
                'Confidence': result['confidence']
            })
        df = pd.DataFrame(results)
        df.to_csv(self.output_path, index=False)

    def get_summary_statistics(self):
        """
        Generates summary statistics from the analysis results, including the total number of questions,
        the distribution of results, and the average confidence score.

        :return: A dictionary containing the summary statistics.
        """
        df = pd.read_csv(self.output_path)
        stats = {
            'total_questions': len(df),
            'results_distribution': df['Result'].value_counts().to_dict(),
            'average_confidence': df['Confidence'].mean(),
        }
        return stats
