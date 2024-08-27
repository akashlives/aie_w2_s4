from langchain_experimental.text_splitter import SemanticChunker
from typing import List


class SemanticChunking:
    def __init__(self, embedding_model, breakpoint_threshold_type="percentile"):
        """
        Initialize the SemanticChunking class with a specified breakpoint threshold type and an embedding model.

        Args:
            embedding_model: An instance of the EmbeddingModel class to generate embeddings.
            breakpoint_threshold_type (str): The type of breakpoint threshold to use for chunking.
                                             Options include 'percentile', 'standard_deviation', 'interquartile'.
        """
        self.text_splitter = SemanticChunker(
            embedding_model, breakpoint_threshold_type=breakpoint_threshold_type
        )

    def split_text(self, text: str) -> List:
        """
        Split the provided text into semantic chunks.

        Args:
            text (str): The text to be split into chunks.

        Returns:
            list: A list of documents (chunks) obtained from the text.
        """
        docs = self.text_splitter.create_documents([text])
        return docs

    def get_chunk(self, docs: List, chunk_index: int) -> str:
        """
        Get a specific chunk from the list of documents.

        Args:
            docs (list): The list of documents (chunks).
            chunk_index (int): The index of the chunk to retrieve.

        Returns:
            str: The content of the specified chunk.
        """
        if chunk_index < 0 or chunk_index >= len(docs):
            raise ValueError(
                f"Chunk index out of range. Please choose a value between 0 and {len(docs) - 1}."
            )

        return docs[chunk_index].page_content

    def get_total_chunks(self, docs: List) -> int:
        """
        Get the total number of chunks in the list of documents.

        Args:
            docs (list): The list of documents (chunks).

        Returns:
            int: The total number of chunks.
        """
        return len(docs)
