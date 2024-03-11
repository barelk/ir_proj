from flask import Flask, request, jsonify
from BM25 import *
from CosineSimilarity import *
from Tokenizer import Tokenizer
from google.cloud import storage
import pickle
from search_backend import *
bucket_name = "assignment3_b"
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
      self.tokenizer = Tokenizer()
      client = storage.Client()
      self.my_bucket = client.bucket(bucket_name=bucket_name)
      self.page_rank = {}

      for blob in client.list_blobs(bucket_name):
            if blob.name == "postings_gcp/index_text.pkl":
                with blob.open('rb') as openfile:
                    self.index_text = pickle.load(openfile)

            elif blob.name == "postings_gcp/index_title.pkl":
                with blob.open('rb') as openfile:
                    self.index_title = pickle.load(openfile)

            # elif blob.name == "postings_gcp/index_anchor.pkl":
            #     with blob.open('rb') as openfile:
            #         self.index_anchor = pickle.load(openfile)

            elif blob.name == "postings_gcp/doc_lengths_title.pkl":
                with blob.open('rb') as openfile:
                    self.doc_lengths_title = pickle.load(openfile)

            elif blob.name == "postings_gcp/doc_lengths_text.pkl":
                with blob.open('rb') as openfile:
                    self.doc_lengths_text = pickle.load(openfile)

            # elif blob.name == "postings_gcp/doc_lengths_anchor.pkl":
            #     with blob.open('rb') as openfile:
            #         self.doc_lengths_anchor = pickle.load(openfile)

            elif blob.name == "postings_gcp/doc_norm_title.pkl":
                with blob.open('rb') as openfile:
                    self.doc_norm_title = pickle.load(openfile)
            
            # elif blob.name == "postings_gcp/doc_norm_text.pkl":
            #     with blob.open('rb') as openfile:
            #         self.doc_norm_text = pickle.load(openfile)

            # elif blob.name == "postings_gcp/doc_norm_anchor.pkl":
            #     with blob.open('rb') as openfile:
            #         self.doc_norm_anchor = pickle.load(openfile)                    
            
            elif blob.name == "postings_gcp/wikiId_title.pkl":
                with blob.open('rb') as openfile:
                    self.wikiId_title = pickle.load(openfile)
                    
            elif blob.name == "postings_gcp/text_idf.pkl":
                with blob.open('rb') as openfile:
                    self.text_idf = pickle.load(openfile)
                    
            elif blob.name == "postings_gcp/title_idf.pkl":
                with blob.open('rb') as openfile:
                    self.title_idf = pickle.load(openfile)

            # elif blob.name == "postings_gcp/anchor_idf.pkl":
            #     with blob.open('rb') as openfile:
            #         self.anchor_idf = pickle.load(openfile)
                     
            elif blob.name == "postings_gcp/pagerank.pkl":
                with blob.open('rb') as openfile:
                    self.pagerank = pickle.load(openfile)
            
            elif blob.name == "postings_gcp/pageviews.pkl":
                with blob.open('rb') as openfile:
                    self.pageviews = pickle.load(openfile)                    
            
      self.cosine_title = CosineSimilarity(self.index_title, bucket_name, self.doc_lengths_title, self.doc_norm_title,self.title_idf)
      #self.cosine_anchor = CosineSimilarity(self.index_anchor, bucket_name, self.doc_lengths_anchor, self.doc_norm_anchor,self.anchor_idf)
      self.BM25_text = BM25(self.index_text, bucket_name, self.doc_lengths_text, None ,self.text_idf)
      #self.BM25_anchor = BM25(self.index_anchor, bucket_name, self.doc_lengths_anchor,None ,self.anchor_idf)
      super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    query_porter_stemmed = list(set(app.tokenizer.tokenize(query, True)))


    query_postinglist_dict_title = get_posinglists_query(app.cosine_title.index, bucket_name, query_porter_stemmed)
    title_res = app.cosine_title.calculate_cosine_similarity(query_porter_stemmed,query_postinglist_dict_title, 100)
    

    query_postinglist_dict_text = get_posinglists_query(app.BM25_text.index, bucket_name, query_porter_stemmed)
    text_res = app.BM25_text.calculate_bm25_v2(query_porter_stemmed, query_postinglist_dict_text, 100) #Number ?


    #query_postinglist_dict_anchor = get_posinglists_query(app.cosine_anchor.index, bucket_name, query_porter_stemmed)
    #anchor_res = app.cosine_anchor.calculate_cosine_similarity(query_porter_stemmed,query_postinglist_dict_anchor, 100)
    #anchor_res = app.BM25_anchor.calculate_bm25_v2(query_porter_stemmed,query_postinglist_dict_anchor, 100)
    
    merge_res = merge_results(title_res,text_res,0.5,0.3,page_rank=app.pagerank,page_rank_weight=0.1, page_views=app.pageviews, page_views_weight=0.1, N=100)

    full_res = get_top_N_ids_titles(merge_res,app.wikiId_title,30)


    # END SOLUTION
    return jsonify(full_res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    
    