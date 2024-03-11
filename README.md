**Project README**

**Introduction**

This project is a search engine implemented in Python using Flask, designed to retrieve relevant documents based on string queries. It utilizes two major retrieval algorithms,such BM25 and Cosine Similarity, to provide search results. The project also integrates with Google Cloud Platform (GCP) for storage and inverted index management.


**Code Structure**

The project consists of several major components:

		BM25 Class (BM25.py): This class implements the Best Match 25 algorithm for document retrieval.
                It calculates the BM25 score for documents based on query terms and document statistics.

		CosineSimilarity Class (CosineSimilarity.py): This class calculates cosine similarity between a query and documents, considering the vector representation of documents.

		Search Fronted (search_frontend.py): The Flask application serves as the core of the search engine. It handles incoming HTTP requests and returns search results.
                The application utilizes BM25 and CosineSimilarity classes for search operations.

		Tokenizer (Tokenizer.py): This module provides tokenization capabilities for text processing. It is used for tokenizing queries and document text.

		Inverted Index (inverted_index_gcp.py): This module manages the inverted index stored in Google Cloud Platform. It provides functionality for reading the index and 
                associated metadata.

		Search Backend (search_backend.py): This module contains utility functions for search operations, including merging search results and retrieving top-ranked documents.


**Functionality**

BM25 Class

	•	__init__(self, index, bucket_name, doc_length, page_rank, idf, k1=1.2, k3=1.2, b=0.75): Initializes the BM25 algorithm with necessary parameters.
	•	calculate_bm25_v2(self, query, query_postinglist_dict, N=100): Calculates BM25 scores for documents relevant to the given query.
CosineSimilarity Class

	•	__init__(self, index, bucket_name, doc_length, doc_vector_normalized, idf): Initializes the Cosine Similarity calculator with required parameters.
	•	calculate_cosine_similarity(self, query, query_postinglist_dict, N=100): Computes cosine similarity scores between the query and documents.
Search Fronte 

	•	search(): Handles the /search route for issuing search queries. It processes the query, calculates search scores using BM25 and Cosine Similarity, and returns the top-ranked documents.


**Usage**

	•	To start the Flask application, run app.py. The server will be publicly available at http://YOUR_SERVER_DOMAIN:8080.
	•	Issue search queries by navigating to /search route with a query parameter, e.g., http://YOUR_SERVER_DOMAIN:8080/search?query=hello+world.


**Dependencies**

	•	Flask: For building the web application.
	•	numpy: For numerical computing.
	•	google-cloud-storage: For interacting with Google Cloud Storage.
	•	pickle: For serializing and deserializing Python objects.
	•	collections: For specialized container datatypes.
	•	math: For mathematical operations.

