from splitingdata import Chunking
from Neural_search import Qdrant_VB, BGE_Embedder, SBERT_Embedder


def save_new_embedding_to_qdrant(api_key, qdrant_url, embedder="BGE_Embedder", collection_name="Data_collection"):
    # initiate chuncker
    custom_transformer = Chunking()
    df = custom_transformer.read_data("data.csv")
    transformed_df = custom_transformer.fit_transform(df)
    documents = custom_transformer.spliting(transformed_df)

    print("creating Embedderr and Qdrant object ")

    ### usage
    qdrant_object = Qdrant_VB(api_key, qdrant_url, collection_name, 1024)
    qdrant_object.create_qdrant_collection_if_not_exist()
    print("Qdrant object created")
    if embedder == "BGE_Embedder":
        embedder = BGE_Embedder()
    else:
        embedder = SBERT_Embedder()
    print("BGE created")
    counter = 0
    for doc in documents:
        print(doc)
        counter += 1
        document_embedding = embedder.create_embedding_for_document(doc)
        qdrant_object.add_embedding_to_qdrant(document_embedding, doc, counter)
        print("doc", doc, " added successfully")


def retrieve_most_similar_from_qdrant(api_key, qdrant_url, embedder="BGE_Embedder", collection_name="Data_collection",
                                      query="", limit=10):
    print("creating Embedderr and Qdrant object ")

    ### usage
    qdrant_object = Qdrant_VB(api_key, qdrant_url, collection_name, 1024)
    print("Qdrant object created")
    if embedder == "BGE_Embedder":
        embedder = BGE_Embedder()
    else:
        embedder = SBERT_Embedder()
    document_embedding = embedder.create_embedding_for_document(query)
    #print("document_embedding",document_embedding)
    similar_vectors = qdrant_object.search_similar_vectors(document_embedding, limit=limit)
    #print("similar_vectors",similar_vectors)
    similar_ids = [result.id for result in similar_vectors]
    similar_texts = qdrant_object.retrieve_text_by_ids(similar_ids)
    return similar_texts


def retrieve_most_similar(type="qdrant", query=""):
    if type == "qdrant":
        api_key = "mt0L253URDvlCcmxCTEzmbjuHnKlFpI9zmB3URtsMwpd6OUuk2aM3Q"
        qdrant_url = "https://9620ee82-03f9-4f04-a642-b006b25e2c7c.us-east4-0.gcp.cloud.qdrant.io"

        similar_texts = retrieve_most_similar_from_qdrant(api_key, qdrant_url, embedder="BGE_Embedder",
                                                          collection_name="Data_collection",
                                                          query=query, limit=10)

        return similar_texts



similar_texts = retrieve_most_similar(type="qdrant", query="Learn how to design large-scale systems")

print(similar_texts)
