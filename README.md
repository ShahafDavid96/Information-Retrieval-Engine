# Information-Retrieval-Engine

title - Search Engine on Wikipedia
In this project we implimented a search engine that searches for relevente wikipedia pages given a query.

code structure:
search_frontend.py
search_backend.py
GCP_handler.py
inverted_index_gcp.py

files that were saved:
PageRank.pickle - holds the page rank for each document
PageViews.pickle - hold the page view of document at a given date
Doc_nom.pickle - hold for each document its norma. The norma is defined on the tf-idf score.

functions:
search - This function uses bm25 on the body and text and we use weights on the body, title, pagerank and pageview to estimate the relevante docs.
search_body - This function uses tf-idf and cosin similarity and retrevice relevante documents only based on the text-index.
search_title - This function uses binary search on the title-index.
search_anchor - This function uses binary search on the anchor-index.
get_page_rank-  This function returns list of page rank values for a given docs list
get_page_view- This function returns list of page views values for a given docs list
