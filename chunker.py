from sentence_transformers import SentenceTransformer
import numpy as np

class RecursiveTextChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=512):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size

    def split_text(self, text):
        """
        Recursively splits text into  chunks of sentences.

        :param text: Raw input text
        :return: List of chunks (strings)
        """
        sentences = self._split_into_sentences(text)
        sentence_embeddings = self.model.encode(sentences)
        chunks = self._create_chunks(sentences, sentence_embeddings)

        return chunks

    def _split_into_sentences(self, text):
        """
        Split text into sentences based on newline characters.

        :param text: Raw input text
        :return: List of sentences
        """
        sentences = text.split("\n")
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunks(self, sentences, embeddings):
        """
        Recursively create chunks that do not exceed the chunk_size.

        :param sentences: List of sentences from the text
        :param embeddings: Embeddings for each sentence
        :return: List of semantic chunks
        """
        chunks = []
        current_chunk = []
        current_chunk_embedding = []

        for i, sentence in enumerate(sentences):
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_chunk_embedding.append(embeddings[i])

            # Calculate total token length of the chunk
            total_chunk_size = self._calculate_chunk_size(current_chunk)

            # If adding this sentence exceeds the chunk size, create a new chunk
            if total_chunk_size > self.chunk_size:
                chunks.append(" ".join(current_chunk[:-1]))  # Add the last valid chunk
                current_chunk = [sentence]  # Start a new chunk
                current_chunk_embedding = [embeddings[i]]

        # Add any remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _calculate_chunk_size(self, chunk):
        """
        Calculate the total size of a chunk, based on sentence length.
        This method could be refined by considering the number of tokens in each sentence.

        :param chunk: List of sentences (chunk)
        :return: Total character size of the chunk
        """
        return sum(len(sentence.split()) for sentence in chunk)
