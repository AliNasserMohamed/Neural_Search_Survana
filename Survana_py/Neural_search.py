from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client import QdrantClient
from FlagEmbedding import FlagModel
from qdrant_client.http.models import VectorParams, Distance
#from splitingdata import Chunking
import logging
from sentence_transformers import SentenceTransformer
import string
from FlagEmbedding import BGEM3FlagModel
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re

"""#**Clean And Spliting Data**"""


class Chunking(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def read_data(self, filename):
        df = pd.read_csv(filename)
        return df

    def fit(self, X, y=None):
        pass

    def clean_data(self, text):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]')
        text = re.sub(emoji_pattern, '', text)
        #text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'(?<!\w)(?<!\.)[^\w\s]*', '', text)
        return text

    def transform(self, X):

        transformed_X = X.copy()

        transformed_X['description'] = transformed_X['description'].str.lower()
        transformed_X['description'] = transformed_X['description'].apply(self.clean_data)

        return transformed_X["description"].values

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def spliting(self, data):
        All_sentences = []
        for text in data:
            sentences = []
            for sentence in text.split("."):
                if len(sentence) > 1:
                    sentences.append(sentence.strip())
            All_sentences.extend(sentences[0:-1])
        return All_sentences


class SBERT_Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the SBERT embedder with the specified model.
        """
        self.embedder = SentenceTransformer(model_name)
        self.vector_size = self.embedder.get_sentence_embedding_dimension()
        logging.info(f"Model '{model_name}' loaded with vector size {self.vector_size}.")

    @staticmethod
    def preprocess_text(text):
        """
        Normalize and preprocess the text.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def create_embedding_for_document(self, document):
        """
        Create an embedding for a given document using the embedding model.
        """
        document = self.preprocess_text(document)
        embedding = self.embedder.encode(document).tolist()
        logging.info(f"Embedding created for document: {document}")
        return embedding

    def create_embeddings_for_documents(self, documents, batch_size=32):
        """
        Create embeddings for a list of documents, where each document is a list of sentences.
        """
        # Concatenate sentences for each document
        concatenated_documents = [' '.join([self.preprocess_text(sentence) for sentence in doc]) for doc in documents]
        embeddings = self.embedder.encode(concatenated_documents, batch_size=batch_size).tolist()
        logging.info(f"Embeddings created for {len(documents)} documents.")
        return embeddings

    def get_query_vector(self, query):
        """
        Convert a query into a vector using the embedding model.
        """
        query = self.preprocess_text(query)
        query_vector = self.embedder.encode(query).tolist()
        logging.info(f"Query vector created for query: {query}")
        return query_vector


class BGE_Embedder:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", vector_size=1024):
        """
        Initialize the BGE embedder with the specified model.
        """
        print("Uploading model")
        self.embedder = BGEM3FlagModel(model_name, use_fp16=True)
        self.vector_size = vector_size
        logging.info(f"Model '{model_name}' loaded with vector size {self.vector_size}.")

    @staticmethod
    def preprocess_text(text):
        """
        Normalize and preprocess the text.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def create_embedding_for_document(self, document):
        """
        Create an embedding for a given document using the embedding model.
        """
        document = self.preprocess_text(document)
        embedding = self.embedder.encode(document)
        logging.info(f"Embedding created for document: {document}")
        return embedding

    def create_embeddings_for_documents(self, documents, batch_size=32):
        """
        Create embeddings for a list of documents, where each document is a list of sentences.
        """
        # Concatenate sentences for each document
        concatenated_documents = [' '.join([self.preprocess_text(sentence) for sentence in doc]) for doc in documents]
        embeddings = self.embedder.encode(concatenated_documents, batch_size=batch_size).tolist()
        logging.info(f"Embeddings created for {len(documents)} documents.")
        return embeddings

    def get_query_vector(self, query):
        """
        Convert a query into a vector using the embedding model.
        """
        query = self.preprocess_text(query)
        query_vector = self.embedder.encode(query).tolist()
        logging.info(f"Query vector created for query: {query}")
        return query_vector


class Qdrant_VB:
    def __init__(self, api_key, qdrant_url, collection_name, vector_size):
        self.api_key = api_key
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key)
        self.vector_size = vector_size

    def create_qdrant_collection_if_not_exist(self):
        """
        Create a Qdrant collection if it does not already exist.
        """
        try:
            if not self.qdrant_client.collection_exists(self.collection_name):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                logging.info(f"Collection '{self.collection_name}' created.")
            else:
                logging.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logging.error(f"Error creating collection '{self.collection_name}': {e}")

    def add_embedding_to_qdrant(self, embedding_vector, document, doc_id):
        """
        Add an embedding vector and corresponding document information to a Qdrant collection.
        """
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=doc_id,
                    vector=embedding_vector,
                    payload={"text": document}
                )]
            )
            logging.info(f"Document ID {doc_id} added to collection '{self.collection_name}'.")
        except Exception as e:
            logging.error(f"Failed to add document ID {doc_id} to collection '{self.collection_name}': {e}")

    def search_similar_vectors(self, query_vector, limit=5):
        """
        Search for vectors in Qdrant that are similar to the query vector.
        """
        try:
            return self.qdrant_client.search(
                collection_name=self.collection_name, query_vector=query_vector, limit=limit
            )
        except Exception as e:
            logging.error(f"Error searching for similar vectors: {e}")
            return []

    def retrieve_text_by_ids(self, ids):
        """
        Retrieve the text of documents from Qdrant given their IDs.
        """
        try:
            results = self.qdrant_client.retrieve(collection_name=self.collection_name, ids=ids)
            return [result.payload["text"] for result in results]
        except Exception as e:
            logging.error(f"Error retrieving texts by IDs: {e}")
            return []



