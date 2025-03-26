"""
Enhanced audit analyzer module for the Knowledge Graph RAG system.
This module handles document processing, knowledge base creation, and query analysis.
"""

from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from .knowledge_graph import KnowledgeGraph, KnowledgeNode
from .document_processor import DocumentProcessor
from .config import CHROMA_PATH, SIMILARITY_THRESHOLD, SEMANTIC_WEIGHT, SEQUENTIAL_WEIGHT
from .utils import logger
import numpy as np

class EnhancedAuditAnalyzer:
    """
    EnhancedAuditAnalyzer class for processing documents, building a knowledge base, and analyzing queries.
    """

    def __init__(self, embeddings):
        """
        Initialize the EnhancedAuditAnalyzer with embeddings, knowledge graph, and document processor.

        Args:
            embeddings: The embeddings model to use for generating document embeddings.
        """
        self.vector_store = None
        self.knowledge_graph = KnowledgeGraph()
        self.document_processor = DocumentProcessor()
        self.embeddings = embeddings

    def process_document(self, pdf_path: str, sentence_model) -> List[str]:
        """
        Process a PDF document and return a list of processed text chunks.

        Args:
            pdf_path: The path to the PDF document.
            sentence_model: The sentence model to use for processing the document.

        Returns:
            A list of processed text chunks.
        """
        pages = self.document_processor.load_pdf(pdf_path)
        if not pages:
            return []
        return self.document_processor.process_pages(pages, sentence_model)

    def build_knowledge_base(self, chunks: List[str], source: str):
        """
        Build a knowledge base from the provided text chunks and source.

        Args:
            chunks: A list of text chunks to build the knowledge base from.
            source: The source of the text chunks.
        """
        documents = []
        embeddings_list = []

        try:
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk)
                embeddings_list.append((chunk, embedding))
                doc = Document(page_content=chunk, metadata={"source": source, "chunk_index": i})
                documents.append(doc)
                node = KnowledgeNode(i, chunk, embedding)
                self.knowledge_graph.add_node(node)

            if documents:
                self.vector_store = Chroma.from_documents(
                    documents,
                    embedding=self.embeddings,
                    persist_directory=CHROMA_PATH
                )
            else:
                logger.error("No valid documents were created, skipping vector store creation.")

            self._build_graph_edges_with_precomputed_embeddings(embeddings_list)

        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
            raise

    def _build_graph_edges_with_precomputed_embeddings(self, embeddings: List[Tuple[str, np.ndarray]]):
        """
        Build graph edges using precomputed embeddings.

        Args:
            embeddings: A list of tuples containing text chunks and their corresponding embeddings.
        """
        try:
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    try:
                        similarity = cosine_similarity(
                            [embeddings[i][1]], [embeddings[j][1]]
                        )[0][0]
                        if similarity > SIMILARITY_THRESHOLD:
                            self.knowledge_graph.add_edge(
                                i, j, "semantic_similarity",
                                similarity * SEMANTIC_WEIGHT
                            )
                        if j == i + 1:
                            self.knowledge_graph.add_edge(
                                i, j, "sequential", SEQUENTIAL_WEIGHT
                            )
                    except Exception as e:
                        logger.error(f"Error building edge between chunks {i} and {j}: {e}")
                        continue
            logger.info("Graph edges successfully built.")
        except Exception as e:
            logger.error(f"Error in _build_graph_edges: {e}")
            raise

    def get_relevant_context(self, query: str) -> str:
        """
        Retrieve relevant context for a given query from the vector store.

        Args:
            query: The query string to search for relevant context.

        Returns:
            A string containing the relevant context.
        """
        if not self.vector_store:
            return ""
        results = self.vector_store.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in results)

    def analyze_query(self, query: str) -> Dict:
        """
        Analyze a query using the knowledge base.

        Args:
            query: The query string to analyze.

        Returns:
            Dict containing findings, conclusion, result, and confidence.
        """
        try:
            context = self.get_relevant_context(query)
            if not context:
                return {
                    "findings": "No relevant context found",
                    "conclusion": "Uncertain",
                    "result": "Uncertain",
                    "confidence": 0
                }

            # Create OpenAI client instance
            client = OpenAI()

            # Use the Chat Completion API with proper client instantiation
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an audit analysis assistant that provides structured responses in JSON format."
                    },
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(question=query, content=context)
                    }
                ],
                temperature=0
            )

            if response and response.choices:
                response_text = response.choices[0].message.content.strip()
                return self._parse_response(response_text)

            return {
                "findings": "No response received from OpenAI API",
                "conclusion": "Uncertain",
                "result": "Uncertain",
                "confidence": 0
            }

        except Exception as e:
            logger.error(f"Error in analyze_query: {e}")
            return {
                "findings": f"Error: {str(e)}",
                "conclusion": "Uncertain",
                "result": "Uncertain",
                "confidence": 0
            }

    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse the response from the OpenAI API.

        Args:
            response_text: The response text from the OpenAI API.

        Returns:
            A dictionary containing the parsed response.
        """
        try:
            clean_text = response_text.strip()
            json_match = re.search(r'(\{[\s\S]*\})', clean_text)
            if not json_match:
                logger.warning("No JSON found in response")
                return {
                    "findings": clean_text[:500],
                    "conclusion": "Error parsing response",
                    "result": "Uncertain",
                    "confidence": 0
                }
            result = json.loads(json_match.group(1))
            required_fields = {'findings', 'conclusion', 'result', 'confidence'}
            if not all(field in result for field in required_fields):
                missing_fields = required_fields - set(result.keys())
                logger.warning(f"Missing required fields in response: {missing_fields}")
                return {
                    "findings": "Missing required fields in response",
                    "conclusion": "Error parsing response",
                    "result": "Uncertain",
                    "confidence": 0
                }
            valid_results = {'Pass', 'Fail', 'NA', 'Uncertain'}
            if result['result'] not in valid_results:
                logger.warning(f"Invalid result value: {result['result']}")
                result['result'] = 'Uncertain'
            try:
                confidence = float(result['confidence'])
                result['confidence'] = max(0, min(100, confidence))
            except (ValueError, TypeError):
                logger.warning("Invalid confidence value")
                result['confidence'] = 0
            return result
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "findings": response_text[:500],
                "conclusion": "Error parsing response",
                "result": "Uncertain",
                "confidence": 0
            }