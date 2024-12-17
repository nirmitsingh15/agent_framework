import faiss
import openai
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def add_texts(self, texts):
        """
        Converts texts to embeddings and stores them in a FAISS index.

        :param texts: List of text chunks
        """
        self.texts = texts
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        dimension = embeddings.shape[1]

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        """
        Searches the FAISS index for the most relevant chunks.

        :param query: Input query string
        :param top_k: Number of top results to retrieve
        :return: Top-k text chunks
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.texts[idx] for idx in indices[0] if idx < len(self.texts)]
