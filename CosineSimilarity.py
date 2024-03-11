import math

import numpy as np
from contextlib import closing
from collections import Counter
from collections import defaultdict
from inverted_index_gcp import MultiFileReader

class CosineSimilarity:
    """
    This class represents a cosine similarity calculator for document retrieval.

    Parameters:
    ----------
    index : dict
        Inverted index containing term-document mappings.
    doc_length : dict
        Dictionary mapping document IDs to their lengths.
    doc_vector_normalized : dict
        Dictionary mapping document IDs to their normalized vectors.
    idf : dict
        Dictionary containing inverse document frequency values for terms.
    """
    def __init__(self, index, bucket_name, doc_length, doc_vector_normalized, idf):
        self.index = index
        self.doc_length = doc_length
        self.doc_vector_normalized = doc_vector_normalized
        self.idf = idf
        self.N = len(doc_length)  # total number of documents
        self.bucket_name = bucket_name

    def calculate_cosine_similarity(self, query,query_postinglist_dict, N=100):
        """
        This function calculates the cosine similarity between the given query and documents.

        Parameters:
        ----------
        query : list of str List of terms in the query.
        query_postinglist_dict : Relevant postings list per query tokens.

        N : int, optional
            Number of documents to return, default is 100.
        
        Returns:
        -------
        sorted list of tuple
            A list of tuples where each tuple contains document ID and its corresponding cosine similarity score.
        """

        terms = tuple(query_postinglist_dict.keys())
        postinglists = tuple(query_postinglist_dict.values())
        query_len = len(np.unique(query))
        query_counter = Counter(query)  # tf
        query_tfidf_dict = self.cal_query_tfidf(query)
        sum = 0
        for term, tfidf in query_tfidf_dict.items():
            sum += (tfidf**2)
        query_vector_normalized = math.sqrt(sum)
        cosine_similarity_dict = defaultdict(float)
        for token in np.unique(query):
            if token not in terms:
              continue
            doc_postinglist = postinglists[terms.index(token)]
            for doc_id, freq in doc_postinglist:
                numerator = ((query_counter[token]/query_len) * self.idf[token]) * ((freq/self.doc_length[doc_id]) * self.idf[token])
                denumerator = query_vector_normalized * self.doc_vector_normalized[doc_id]
                if doc_id in cosine_similarity_dict.keys():
                    cosine_similarity_dict[doc_id] += numerator / denumerator
                else:
                    cosine_similarity_dict[doc_id] = numerator / denumerator

        sorted_cos_sim_list = sorted(cosine_similarity_dict.items(), key=lambda item: item[1] , reverse=True)
        return sorted_cos_sim_list [:N]

    def cal_query_tfidf(self, query):
        """
        This function calculate the tf-idf of the query
        :param query:
        :return:
        """
        tfidf_query_dict = {}
        query_len = len(np.unique(query))
        query_counter = Counter(query)  # tf
        for term in np.unique(query):
            if term in self.index.df.keys():
                tfidf_query_dict[term] = (query_counter[term]/query_len) * self.idf[term]
        return tfidf_query_dict