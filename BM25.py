import math
import numpy as np
from contextlib import closing
from collections import Counter
from inverted_index_gcp import MultiFileReader

class BM25:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.2
    k3 : float, default 1.2
    b : float, default 0.75
    """

    def __init__(self, index, bucket_name, doc_length, page_rank, idf, k1=1.2, k3=1.2, b=0.75):
        self.index = index
        self.doc_length = doc_length
        self.page_rank = page_rank
        self.b = b
        self.k1 = k1
        self.k3 = k3
        self.N = len(doc_length)  # total number of documents
        self.avg_doc_length = sum(doc_length.values()) / self.N
        self.idf = idf
        self.bucket_name = bucket_name


    def normalize_bm25_scores(self, bm25_scores):
        """
        Normalize BM25 scores using min-max normalization.
        Args:
            bm25_scores (list): List of BM25 scores for documents.
        Returns:
            list: Normalized BM25 scores.
        """
        scores = [score for _, score in bm25_scores]
        min_score = min(scores)
        max_score = max(scores)
        normalized_scores = []
        for doc_id, score in bm25_scores:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_scores.append((doc_id, normalized_score))
        return normalized_scores

    def calculate_bm25_v2(self, query,query_postinglist_dict, N=100):
        """
        This function calculates the BM25 score for each document relevant to the given query.
        
        Parameters:
        ----------
        query : list of str List of terms in the query.
        query_postinglist_dict : Relevant postings list per query tokens.
        N : int, optional
            Number of documents to return, default is 100.

        Returns:
        -------
        sorted list
            Tuples, where keys are document IDs and values are their corresponding BM25 scores.
        """

        scores = {}
        #query_postinglist_dict = self.get_posinglists_query(query)
        query_counter = Counter(query)  # query term frequency
        for term, posting_list in query_postinglist_dict.items():
            idf_bm = math.log10((self.N + 1) / self.index.df.get(term, 0))  # Handle term not found
            for doc_id, tf in posting_list:
                doc_length = self.doc_length.get(doc_id, 0)  # Handle doc_id not found
                tf_weight_doc = ((self.k1 + 1) * tf) / (self.k1 * ((1 - self.b) + self.b * (doc_length / self.avg_doc_length)) + tf)  # Handle zero tf?
                tf_weight_query = ((self.k3 + 1) * query_counter[term]) / (self.k3 + query_counter[term])
                score = idf_bm * tf_weight_doc * tf_weight_query
                scores[doc_id] = scores.get(doc_id, 0) + score

        norm_bm25 = self.normalize_bm25_scores(scores.items())
        return  sorted(norm_bm25, key=lambda item: item[1], reverse=True) [:N]