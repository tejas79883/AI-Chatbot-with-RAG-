# vectors.py

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

import os


class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def create_embeddings(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No documents were loaded from the PDF.")

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        if not splits:
            raise ValueError("No text chunks were created from the documents.")

        try:
            # Initialize Qdrant client
            client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)

            # 🔴 DELETE old collection (important)
            collections = [c.name for c in client.get_collections().collections]
            if self.collection_name in collections:
                client.delete_collection(collection_name=self.collection_name)

            # ✅ Create fresh collection
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )
            # Connect LangChain Qdrant wrapper
            qdrant = Qdrant(
                client=client,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )

            # Add documents
            qdrant.add_documents(splits)

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        return "✅ Vector DB Successfully Created and Stored in Qdrant!"