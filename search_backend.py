from inverted_index_gcp import MultiFileReader


def merge_results(title_scores, text_scores, title_weight=0.5, text_weight=0.3, page_rank=None, page_rank_weight=0.1, page_views=None, page_views_weight=0.1, N=100):
    """
    This function merge and sort documents retrieved by its weighte score
    Parameters:
    -----------
    title_scores: a dictionary of key: doc_id ,value: title index similarity scores. format:(doc_id,score).
                                                                         
    text_scores: a dictionary of key: doc_id ,value: text index similarity scores format:(doc_id,score).

    anchor_scores: a dictionary of key: doc_id ,value: anchor index similarity scores format:(doc_id,score).

    page_rank : page rank algorithm results format:(doc_id,score).

    title_weight: float, scores for weigted title scores
    text_weight: float, scores for weigted text scores
    anchor_weight: float, for weigted anchor scores

    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.
    Returns:
    -----------
    list of topN pairs:
                    key: doc_id
                    value: merged scores format:(doc_id,score).
    """
    merged_results = []

    max_page_rank = page_rank.get('3434750', 1)  # Extract the maximum page rank value for normalization
    max_page_views = page_views.get('15580374', 1)

    title_dict = {doc_id: [score * title_weight] for doc_id, score in title_scores}
    text_dict = {doc_id: [score * text_weight] for doc_id, score in text_scores}

    common_doc_ids = set(title_dict.keys()) & set(text_dict.keys()) 

    different_doc_ids = (set(title_dict.keys()) ^ set(text_dict.keys()))

    for doc_id in different_doc_ids:
        if doc_id in title_dict:
            merged_results.append((doc_id, title_dict[doc_id][0]))
        else:
            merged_results.append((doc_id, text_dict[doc_id][0]))


    for doc_id in common_doc_ids:
        title_score = title_dict.get(doc_id, 0)
        body_score = text_dict.get(doc_id, 0)
        
        comb_score = title_score[0] + body_score[0]
       
        merged_results.append((doc_id, comb_score))

    # Normalize scores if page_rank is provided and give weight to it
    if page_rank:
        for doc_id in merged_results:
            if doc_id in page_rank:
                merged_results[doc_id] += (page_rank_weight * page_rank[doc_id] / max_page_rank)

    # Normalize scores if page_views is provided and give weight to it
    if page_views:
        for doc_id in merged_results:
            if doc_id in page_views:
                merged_results[doc_id] += (page_views_weight * page_views[doc_id] / max_page_views)

    # Sort merged results by score in descending order and return top N
    merged_results.sort(key=lambda x: round(x[1],5), reverse=True)
    return merged_results[:N]


def get_top_N_ids_titles(merged_results,wikiId_title, N=100):
    """
    Retrieve the top N document IDs and their corresponding titles from merged results.

    Parameters:
    ----------
        merged_results (list): List of tuples containing document IDs and scores.
        wikiId_title (dict): Dictionary mapping document IDs to titles.
        N (int, optional): Number of top documents to retrieve. Defaults to 100.

    Returns:
    -------
        list: List of tuples containing document IDs and titles of the top N documents.
    """
    return [(str(id),wikiId_title[id]) for (id,score) in merged_results] [:N]

def get_posinglists_query(index, bucket_name, query):
    """
    Retrieve relevant posting lists for terms in the query.

    Parameters:
    ----------
        index (Index): The inverted index and document frequency information.
        bucket_name (str): Name of the document storage bucket.
        query (list): List of terms in the query.

    Returns:
    -------
        dict: A dictionary where keys are terms in the query and values are their respective posting lists.
    """
    postinglist_dict = {}
    for term in query:
        if index.df.get(term) is not None:
            temp = index.read_a_posting_list('.',term,bucket_name)
            postinglist_dict[term] = temp
    return postinglist_dict
'''
def get_doc_list_pageRank(index, ids, N, pageRank):
    pageRank_dict = {}
    for id in ids:
        pageRank_dict[id] = pageRank[id]
    return pageRank_dict

'''