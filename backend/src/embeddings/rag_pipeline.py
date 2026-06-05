"""RAG (Retrieval-Augmented Generation) pipeline."""

import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for document retrieval and processing."""

    def __init__(self, embedding_service: EmbeddingService, chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """Initialize RAG pipeline."""
        self.embedding_service = embedding_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info("RAGPipeline initialized")

    def process_documents(self, documents: List[dict]) -> None:
        """Process and index documents."""
        logger.info(f"Processing {len(documents)} documents")

        doc_list = [
            Document(
                page_content=f"{doc.get('name', '')}\n{doc.get('description', '')}\n{doc.get('ingredients', '')}",
                metadata={
                    "id": doc.get("id"),
                    "name": doc.get("name"),
                    "price": doc.get("price"),
                    "url": doc.get("url"),
                },
            )
            for doc in documents
        ]

        split_docs = self.text_splitter.split_documents(doc_list)
        logger.info(f"Split into {len(split_docs)} chunks")

        texts = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]

        self.embedding_service.add_documents(texts, metadatas)

    def retrieve(self, query: str, k: int = 5) -> List:
        """Retrieve relevant documents."""
        logger.info(f"Retrieving {k} documents for query: {query}")
        results = self.embedding_service.similarity_search(query, k=k)
        return results
