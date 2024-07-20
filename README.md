# Neural_Search_Survana
### Survana Neural Search
An advanced search engine powered by neural networks to enhance search relevance and accuracy. Leverages deep learning to parse and understand complex queries across large datasets.

Description of the Implementation for Data Cleaning, Embedding, and Vector Search using Qdrant

1- Data Cleaning and Splitting
Chunking Class

●	Purpose: Clean and split textual data into manageable chunks.

●	Methods:

○	read_data: Reads a CSV file into a DataFrame.

○	clean_data: Removes URLs, emojis, and special characters from text.

○	transform: Converts text to lowercase and applies clean_data method to the 'description' column.

○	spliting: Splits text data into sentences and returns a list of sentences.

2- Embedding Models

SBERT_Embedder Class

●	Purpose: Generate embeddings for documents using the Sentence-BERT model.

●	Initialization: Loads the specified Sentence-BERT model and determines the vector size.

●	Methods:

○	preprocess_text: Converts text to lowercase and removes punctuation.

○	create_embedding_for_document: Generates an embedding for a single document.

○	create_embeddings_for_documents: Generates embeddings for a list of documents, concatenating sentences within each document.

○	get_query_vector: Converts a query into an embedding vector.

BGE_Embedder Class

●	Purpose: Generate embeddings using the BGE (BAAI/bge-reranker-v2-m3) model.

●	Initialization: Loads the specified BGE model and determines the vector size.

●	Methods:

○	preprocess_text: Converts text to lowercase and removes punctuation.

○	create_embedding_for_document: Generates an embedding for a single document.

○	create_embeddings_for_documents: Generates embeddings for a list of documents, concatenating sentences within each document.

○	get_query_vector: Converts a query into an embedding vector.

3- Vector Search with Qdrant

Qdrant_VB Class
●	Purpose: Manage a Qdrant collection for storing and searching document embeddings.

●	Initialization: Initializes Qdrant client with API key, URL, collection name, and vector size.
●	Methods:

○	create_qdrant_collection_if_not_exist: Checks if a Qdrant collection exists; if not, it creates a new one with specified vector parameters.

○	add_embedding_to_qdrant: Adds an embedding vector and corresponding document to the Qdrant collection.

○	search_similar_vectors: Searches for vectors in Qdrant that are similar to a given query vector.

○	retrieve_text_by_ids: Retrieves the text of documents from Qdrant given their IDs.

4- Usage Flow
1.	Data Reading and Cleaning:
2.	
○	Read the CSV file containing the textual data using Chunking.read_data.

○	Clean the text data using Chunking.transform.

○	Split the cleaned text into sentences using Chunking.spliting.

4.	Embedding Generation:
5.	
○	Initialize either SBERT_Embedder or BGE_Embedder with the chosen model.

○	Preprocess and create embeddings for the text data using the methods provided in the embedder classes.

7.	Qdrant Collection Management:
   
○	Initialize Qdrant_VB with necessary parameters.

○	Create a Qdrant collection if it doesn't exist using create_qdrant_collection_if_not_exist.

○	Add document embeddings to Qdrant using add_embedding_to_qdrant.

9.	Vector Search:
○	Search for similar vectors in Qdrant using search_similar_vectors.


○	Retrieve document texts from Qdrant using retrieve_text_by_ids.





5- Functions for Integration

save_new_embedding_to_qdrant

●	Purpose: Clean data, generate embeddings, and save them to Qdrant.

●	Parameters: API key, Qdrant URL, embedder type, collection name.

●	Process:

○	Read and clean data using Chunking.

○	Split data into sentences.

○	Initialize Qdrant_VB and check/create collection.

○	Initialize embedder (BGE or SBERT).

○	Generate embeddings and add them to Qdrant.

retrieve_most_similar_from_qdrant

●	Purpose: Retrieve the most similar documents to a query from Qdrant.

●	Parameters: API key, Qdrant URL, embedder type, collection name, query, limit.

●	Process:

○	Initialize Qdrant_VB.

○	Initialize embedder (BGE or SBERT).

○	Generate embedding for query.

○	Search for similar vectors in Qdrant.

○	Retrieve and return the text of similar documents.

retrieve_most_similar

●	Purpose: High-level function to retrieve similar documents, abstracting the backend.
●	Parameters: Type (e.g., "qdrant"), query.
●	Process:
○	Define Qdrant API key and URL.
○	Call retrieve_most_similar_from_qdrant with parameters.
○	Return the similar texts.


Requirements libraries  :
transformers
qdrant-client
torch
FlagEmbedding
sentence_transformers
sklearn
