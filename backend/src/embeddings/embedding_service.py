"""Embedding service using OpenAI."""

import logging
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", 
                 chromadb_path: str = "./data/chromadb", collection_name: str = "king_arthur"):
        """Initialize embedding service."""
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=model,
        )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=chromadb_path,
        )
        logger.info("EmbeddingService initialized")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        try:
            embeddings = self.embeddings.embed_documents(documents)
            logger.info(f"Generated embeddings for {len(documents)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def add_documents(self, documents: List[str], metadatas: List[dict] = None) -> None:
        """Add documents to vector store."""
        try:
            self.vectorstore.add_texts(documents, metadatas)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5):
        """Search for similar documents."""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching similarity: {e}")
            raise
