import numpy as np
from inverted_index_gcp import InvertedIndex, MultiFileReader
from GCP_handler import ReadFromGcp
from collections import defaultdict, Counter
import re
import math
import nltk
import time
from nltk.corpus import stopwords

import pickle
import gensim
import gensim.downloader


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))

epsilon = 0.0000001




def tokenize(text):
    """
        This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

        Parameters:
        -----------
        text: string , represting the text to tokenize.

        Returns:
        -----------
        list of tokens (e.g., list of tokens).
        """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens


class SearchHandler:
    """
    Class that handles the searches over the indexes

    ...

    Attributes
    ----------
    handler : ReadFromGcp obj
    inverted_index : InvertedIndex object

    Methods
    -------
    search_body(q):
        Function that returns the best 100 documents
    """

    def __init__(self):
        self.handler = ReadFromGcp("316233154")
        self.inverted_index = {
            "body": self.handler.get_inverted_index(source_idx=f"postings_gcp_text/index.pkl",
                                                    dest_file=f"text_index.pkl"),
            "title": self.handler.get_inverted_index(source_idx=f"postings_gcp_title/index.pkl",
                                                     dest_file=f"title_index.pkl"),
            "titles_dict": self.handler.load_pickle_file(source=f"titles_dict.pickle", dest=f"titles_dict.pkl")
            , "page_rank": self.handler.load_pickle_file(source=f"pageRank.pickle", dest=f"pageRank.pkl")
            , "norm": self.handler.load_pickle_file(source=f"norm_dict.pickle", dest=f"norm_dict.pkl")
            , "page_view": self.handler.load_pickle_file(source=f"page_views.pkl", dest=f"page_views.pkl")
            , "anchor": self.handler.get_inverted_index(source_idx=f"postings_gcp_anchor/index.pkl" ,dest_file = f"anchor_index.pkl")
            # , "idf_with_stem": self.handler.load_pickle_file(source=f"page_views.pkl", dest=f"page_views.pkl")
            # "stem_body": self.handler.get_inverted_index(source_idx=f'postings_gcp_text_with_stemming/index.pkl',dest_file=f"stem_text_index.pkl")
            # "stem_title":self.handler.get_inverted_index(source_idx=f'postings_gcp_title_with_stemming/index.pkl',dest_file=f"stem_title_index.pkl")
        }
        self.idx_body = self.inverted_index["body"].read_index('.', 'text_index')
        self.idx_title = self.inverted_index["title"].read_index('.', 'title_index')
        self.idx_anchor = self.inverted_index["anchor"].read_index('.', 'anchor_index')
        # self.idx_stem_body = self.inverted_index["stem_body"].read_index('.', 'stem_text_index')
        # # self.idx_stem_title = self.inverted_index["stem_title"].read_index('.', 'stem_title_index')
        #
        # self.idx_stem_body.doc_len=self.handler.load_pickle_file(source=f"dl_text_with_stemming.pkl", dest=f"dl_text_with_stemming.pkl")
        max_page = max(self.inverted_index["page_rank"].values())
        max_view = max(self.inverted_index['page_view'].values())

        self.page_rank={k:(v/max_page) for k,v in self.inverted_index["page_rank"].items()}
        self.page_view ={k:(v/max_view) for k,v in self.inverted_index["page_view"].items()}
        # self.DL=self.handler.load_pickle_file(source=f"doc_len.pkl", dest=f"doc_len.pkl")
        self.glove= gensim.downloader.load('glove-wiki-gigaword-300')

    def query_exp(self,query):
        glove_vectors = self.glove
        expanded_tokens = []
        for term in query:
            exp_for_token = glove_vectors.most_similar(term, topn=1)
            for w in exp_for_token:
                if w[1]>0.85:
                    w = tokenize(w[0])
                    expanded_tokens.append(w[0])
        query = query + expanded_tokens
        return query


    def search(self, q):
        query = tokenize(q)
        try:
             query = self.query_exp(query)
        except:
             print("error")

        print(query)
        index = "text"
        text_post = {}
        title_post = {}
        for w in query:
            try:
                text_post[w] = self.handler.read_posting_list(self.idx_body, w, "text")
            except:
                text_post[w] = []
            try:
                title_post[w] = self.handler.read_posting_list(self.idx_title, w, "title")
            except:
                title_post[w] = []

        BM25_text = BM25_from_index(self.idx_body, text_post)
        bm25_text_train = BM25_text.search(query, 100)

        BM25_title = BM25_from_index(self.idx_title, title_post)
        bm25_title_train = BM25_title.search(query, 100)
        page_rank_res = self.get_page_rank([doc_id for doc_id, _ in bm25_text_train])
        page_views_res = self.get_page_view([doc_id for doc_id, _ in bm25_text_train])

        merge = multi_merge_results([bm25_text_train, bm25_title_train,page_rank_res,page_views_res],[0.5, 0.3,0.1,0.1], 20)
        res = []

        res = [(int(key[0]), self.inverted_index["titles_dict"][key[0]]) for key in merge]
        return res

    def search_body(self, q):
        """
        This function returns the 100 documents.

        Parameters
        ----------
        q : str
            The sound the animal makes (default is None)

        Return
        -----
        result : List[Tuples(doc_id, title)]
        """
        # Tokenize the query

        query = tokenize(q)
        dic = defaultdict(list)
        # Read inverted index from the bucket

        tfidf_q = {}
        Q = generate_query_tfidf_vector(query, self.idx_body, self.idx_body.doc_len)

        counter = Counter(query)
        lst_of_tuples = []
        q_norm_sum = 0
        for token in counter:
            q_idf = math.log10(len(self.idx_body.doc_len) / self.idx_body.df.get(token,1))
            token_tfidf = (counter[token] / len(q)) * q_idf
            lst_of_tuples.append((token, token_tfidf))
            q_norm_sum += token_tfidf ** 2

        doc_rankings = {}
        index = "text"
        # Make a dictionary with the term and its posting list
        cos_dict = {}
        for w in query:
            pos_lst = self.handler.read_posting_list(self.idx_body, w, index)
            dic[w] = pos_lst

        # Calculate tf_idf for each document
        for word, post in dic.items():
            tfidf_q[word] = []
            for doc_id, tf in dic[word]:
                tfidf = (tf / self.idx_body.doc_len.get(doc_id, 1 / epsilon)) * math.log(
                    (len(self.idx_body.doc_len)) / self.idx_body.df[word])
                if doc_id not in cos_dict:
                    cos_dict[doc_id] = np.dot(tfidf, Q[word])
                else:
                    cos_dict[doc_id] = cos_dict[doc_id] + np.dot(tfidf, Q[word])
        cos_sim_results = []
        # print(self.inverted_index["norm"])
        for doc_id in cos_dict:
            cos_sim_results.append((doc_id, cos_dict[doc_id] / (self.inverted_index["norm"][doc_id] * q_norm_sum)))
        # result = sorted(
        #     [(doc_id, cos_dict[doc_id] / (self.inverted_index["norm"][doc_id] * q_norm_sum)) for doc_id in cos_dict],
        #     key=lambda x: x[1], reverse=True)
        result = sorted(cos_sim_results, key=lambda x: x[1], reverse=True)

        res = [(key, self.inverted_index["titles_dict"][key]) for key, val in result]

        return res

    def search_title(self, q):
        """
        This function returns the ALL the documents who have one of the query words in thier title.

        Parameters
        ----------
        q : str
            q as a query to search for,
            ex: The sound the animal makes (default is None)

        Return
        -----
        result : List[Tuples(doc_id, title)]
        """
        query = tokenize(q)
        index = "title"
        dic = defaultdict(list)
        # Read inverted index from the bucket
        idx_title = self.idx_title
        dic = defaultdict(list)
        title_ranking = {}

        for w in query:
            dic[w] = self.handler.read_posting_list(idx_title, w, index)

        for key, val in dic.items():
            for doc_id, count in val:
                if doc_id not in title_ranking.keys():
                    title_ranking[doc_id] = 0
                title_ranking[doc_id] = title_ranking[doc_id] + 1

        sorted_d = dict(sorted(title_ranking.items(), key=lambda item: item[1], reverse=True))
        sorted_keys = list(sorted_d.keys())

        res = []
        for key in sorted_keys:
            res.append((key, self.inverted_index["titles_dict"][key]))
        return res

    def search_anchor(self, q):
        """
        This function returns the 100 documents.

        Parameters
        ----------
        q : str
            The sound the animal makes (default is None)

        Return
        -----
        result : List[Tuples(doc_id, title)]
        """
        query = tokenize(q)
        index = "anchor"
        dic = defaultdict(list)
        # Read inverted index from the bucket
        idx_anchor = self.idx_anchor

        dic = defaultdict(list)
        anchor_ranking = {}
        for w in query:
            dic[w] = self.handler.read_posting_list(idx_anchor, w, index)

        for key, val in dic.items():
            for doc_id, count in val:
                if doc_id not in anchor_ranking.keys():
                    anchor_ranking[doc_id] = 0
                anchor_ranking[doc_id] = anchor_ranking[doc_id] + 1

        sorted_d = dict(sorted(anchor_ranking.items(), key=lambda item: item[1], reverse=True))
        sorted_keys = list(sorted_d.keys())
        res = []
        for key in sorted_keys:
            try:
                res.append((key, self.inverted_index["titles_dict"][key]))
            except:
                res.append(key)
        return res

    def get_page_rank(self, docs):
        res = []

        for doc in docs:
            res.append((doc,self.page_rank.get(doc,0)))
        return res

    def get_page_view(self, docs):
        res = []
        for doc in docs:
            res.append((doc,self.page_view.get(doc,0)))
        return res

    def search_config(self, query, config):
        query = tokenize(query)
        text_post = {}
        title_post = {}
        for w in query:
            try:
                text_post[w] = self.handler.read_posting_list(self.idx_body, w, "text")
            except:
                text_post[w] = []
            try:
                title_post[w] = self.handler.read_posting_list(self.idx_title, w, "title")
            except:
                title_post[w] = []

        body_bm25 = BM25_from_index(self.idx_body, text_post, k1=config['body_k'], b=config['body_b'])
        body_res = body_bm25.search(query)
        title_bm25 = BM25_from_index(self.idx_title, title_post, k1=config['title_k'], b=config['title_b'])
        title_res = title_bm25.search(query)

        page_rank_res = self.get_page_rank([doc_id for doc_id, _ in body_res])
        page_views_res = self.get_page_view([doc_id for doc_id, _ in body_res])
        ws = [config['body_w'], config['title_w'], config['page_rank_w'], config['page_views_w']]

        res = multi_merge_results([body_res, title_res, page_rank_res, page_views_res], ws,20)

        return res


def generate_query_tfidf_vector(query_to_search, index, DL):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """
    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    dic = {}
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)

                Q[ind] = tf * idf
                dic[token] = tf * idf
            except:
                pass
    return dic


class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, dic, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.doc_len)
        self.AVGDL = sum(index.doc_len.values()) / self.N
        self.w2pls = dic
        self.term_freq = {}

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        # idf = {}
        idf = {term: math.log(1 + (self.N - self.index.df[term] + 0.5) / (self.index.df[term] + 0.5), 10) for term in
               list_of_tokens if term in self.index.df.keys()}
        # for term in list_of_tokens:
        #     if term in self.index.df.keys():
        #         n_ti = self.index.df[term]
        #         idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5), 10)
        #     else:
        #         pass
        return idf

    def get_candidate_documents_and_scores(self, query_to_search):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = {}
        for term in np.unique(query_to_search):
            self.term_freq[term] = {}
            normlized_tfidf = []
            if term in self.w2pls:
                list_of_doc = self.w2pls[term]
                for doc_id, freq in list_of_doc:
                    normlized_tfidf.append((doc_id, (freq / self.index.doc_len.get(doc_id, 1 / epsilon)) * (
                        math.log((len(self.index.doc_len) / self.index.df.get(term, 1)), 10))))

                    self.term_freq[term][doc_id] = freq

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def search(self, query, N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """

        # candidates = self.get_candidate_documents_and_scores(query)
        # candidates = dict(sorted(candidates.items(), key=lambda x: x[1],reverse=True)[:5000])
        # distinct_candi_ids = np.unique([i[0] for i in candidates])
        candidates = self.get_candidate_docs(query)
        self.idf = self.calc_idf(query)
        scores = sorted([(doc_id, self._score(query, doc_id)) for doc_id in candidates], key=lambda x: x[1],
                        reverse=True)
        return scores

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0

        doc_len = self.index.doc_len.get(doc_id, 1 / epsilon)
        for term in query:
            if term in self.index.df.keys():
                if doc_id in self.term_freq[term]:
                    freq = self.term_freq[term][doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)

        return score

    def get_candidate_docs(self, query):
        res = set()
        for token in np.unique(query):
            if token in self.index.df.keys():
                pl = self.w2pls[token]
                self.term_freq[token] = {}
                for doc_id, tf in pl:
                    self.term_freq[token][doc_id] = tf
                    res.add(doc_id)
        return res


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """

    merged = []
    title_dict = {doc_id: score for doc_id, score in title_scores}
    body_dict = {doc_id: score for doc_id, score in body_scores}

    for k in set(list(title_dict.keys()) + list(body_dict.keys())):
        if k in title_dict.keys() and k in body_dict.keys():
            merged.append((k, title_weight * title_dict[k] + text_weight * body_dict[k]))
        elif k in title_dict.keys():
            merged.append((k, title_weight * title_dict[k]))
        else:
            merged.append((k, text_weight * body_dict[k]))

    merged = sorted(merged, key=lambda tup: tup[1], reverse=True)[:N]
    return merged


def multi_merge_results(scores, weights, N=100):
    merged_scores = {}
    for tuples_list, weight in zip(scores, weights):
        try:
            for id , score in tuples_list:
                if id in merged_scores:
                    merged_scores[id] += score * weight
                else:
                    merged_scores[id] = score * weight
        except:
            print(tuples_list)

    merged_list = [(k, v) for k, v in sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)]

    return merged_list[:N]
