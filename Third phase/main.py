from __future__ import unicode_literals
import math
import operator
from collections import Counter

import pandas as pd
from hazm import *
import numpy as np
from gensim.models import Word2Vec
from numpy.linalg import norm
import multiprocessing
from random import randrange
import copy

k = 10  # number of shown results
K = 15  # KNN
docs_to_read = 11000
docs_to_label = 2000
epochs = 300
cent_count = 7
b = 2


def tf_idf(term_frequency, doc_frequency, N):
    tf = 1 + math.log10(term_frequency)
    idf = math.log10(N / doc_frequency)
    tf_idf_weight = tf * idf

    return tf_idf_weight


def similarity(doc1, doc2):
    score = np.dot(doc1, doc2) / (norm(doc1) * norm(doc2))
    return (score + 1) / 2


def doc_embed(tf_idf_docs, w2v_model):
    docs_embedding = []
    for doc in tf_idf_docs:
        doc_vec = np.zeros(300)
        weight_sum = 0
        for token, weight in doc.items():
            try:
                doc_vec += w2v_model.wv[token] * weight
                weight_sum += weight
            except KeyError:
                True
        docs_embedding.append(doc_vec / weight_sum)
    return docs_embedding


def query_similarity_with_all(doc_list, query_vector):
    results = []
    for document in doc_list:
        results.append(similarity(query_vector, document))
    results = np.array(results)
    results = results.argsort()[-k:][::-1]
    return results


def clustering(doc_vec_list, centroid_list):
    infunc_clustered_docs = []
    for cent in range(cent_count):
        infunc_clustered_docs.append([0])

    for doc_id in range(len(doc_vec_list)):
        best_similarity = 0
        this_doc_cluster = 0
        for i in range(cent_count):
            similarity_with_this_one = similarity(doc_vec_list[doc_id], centroid_list[i])
            if similarity_with_this_one > best_similarity:
                best_similarity = similarity_with_this_one
                this_doc_cluster = i
        infunc_clustered_docs[this_doc_cluster].append(doc_id)
    for i in range(cent_count):
        infunc_clustered_docs[i].pop(0)
    return infunc_clustered_docs


def calculate_centroid(doc_vec_list, cluster_doc_id_list):
    cluster_vec_sum = doc_vec_list[cluster_doc_id_list[0]]
    for cluster_doc_id in cluster_doc_id_list[1:]:
        cluster_vec_sum += doc_vec_list[cluster_doc_id]
    return cluster_vec_sum / len(cluster_doc_id_list)


cores = multiprocessing.cpu_count()

# Reading The Data
dataset0 = [pd.read_excel('IR00_3_11k News.xlsx'), pd.read_excel('IR00_3_17k News.xlsx'),
            pd.read_excel('IR00_3_20k News.xlsx')]
dataset = pd.concat(dataset0, ignore_index=True)
docs = dataset['content']
urls = dataset['url']
docs = docs[:docs_to_read]
N = len(docs)
topics = dataset['topic']

# Preprocessing

# Normalizing
normalizer = Normalizer()
for doc in docs:
    doc = normalizer.normalize(doc)

# Tokenizing and Stemming
token_stream = []
stemmer = Stemmer()

for docid in range(len(docs)):
    terms = word_tokenize(docs[docid])
    for pos in range(len(terms)):
        term = stemmer.stem(terms[pos])
        term = terms[pos]
        token_stream.append([term, [docid, pos]])
token_stream.sort(key=lambda x: x[0])

# index construction
positional_index = [token_stream[0]]

for token in token_stream:
    if token[0] != positional_index[-1][0]:
        positional_index.append(token)
    elif token[1][0] != positional_index[-1][-1][0]:
        positional_index[-1].append(token[1])
    else:
        positional_index[-1][-1].append(token[1][1])
# frequency calculation
term_freq = []
freqs = []
for term in positional_index:
    docs = term[1:]
    freq = 0
    for docid in docs:
        freq += len(docid) - 1
    term_freq.append([freq, term[0]])

term_freq.sort()
term_freq.reverse()

# stop word removal
for i in range(15):
    for term in positional_index:
        if term[0] == term_freq[i][1]:
            positional_index.remove(term)

# Ranked Retrieval
# tf idf docs construction
weight_postings_list = {}
tf_idf_docs = [{} for sub in range(N)]
for term in positional_index:
    docs = term[1:]
    df = len(term) - 1
    weight_postings_list[term[0]] = tf_idf(1, df, N)
    for doc in docs:
        tf = len(doc) - 1
        tf_idf_docs[doc[0]][term[0]] = tf_idf(tf, df, N)

# # training model
# my_w2v_model = Word2Vec(min_count=1, window=5, vector_size=300, alpha=0.03, workers=cores - 1)
# # # make training data
# training_data = []
# for doc in tf_idf_docs:
#     training_data.append(doc.keys())

# # train
# my_w2v_model.build_vocab(training_data)
# my_w2v_model.train(training_data, total_examples=my_w2v_model.corpus_count, epochs=20)
# # docs
# # my_model_docs_embedding = doc_embed(tf_idf_docs, my_w2v_model)
w2v_model = Word2Vec.load("w2v_150k_hazm_300_v2.model")  # given model
given_docs_embedding = doc_embed(tf_idf_docs, w2v_model)

part = input()
if part == "1":

    # make centroids
    centroids = []
    # # random firsts
    for i in range(cent_count):
        centroids.append(given_docs_embedding[randrange(N)])

    # clustering loop
    last_centroids = copy.deepcopy(centroids)
    flag = True
    n = 0
    while flag and n < epochs:
        # print(n)
        n += 1
        clustered_docs = clustering(given_docs_embedding, centroids)  # clustering
        last_centroids = copy.deepcopy(centroids)
        for j in range(cent_count):
            centroids[j] = calculate_centroid(given_docs_embedding, clustered_docs[j])  # update centroids
        flag = False  # check if new centroid are identical to previous
        for cent in range(len(centroids)):
            for item in range(len(centroids[cent])):
                if centroids[cent][item] != last_centroids[cent][item]:
                    flag = True

    # print(clustered_docs[0])
    # print(clustered_docs[1])
    # print(clustered_docs[2])
    # print(clustered_docs[3])
    # print(clustered_docs[4])

    # query processing
    query = input()
    query = normalizer.normalize(query)
    # query = stemmer.stem(query)
    query = word_tokenize(query)
    # q vector constrcution
    q_vector = np.zeros(300)

    weight_sum = 0
    for word in query:
        q_vector += w2v_model.wv[word] * weight_postings_list[word]
        weight_sum += weight_postings_list[word]
    q_vector = q_vector / weight_sum

    similarities = []
    for i in range(len(centroids)):
        similarities.append([i, similarity(q_vector, centroids[i])])
    similarities = sorted(similarities, key=lambda x: (x[1]))
    similarities.reverse()
    clustering_results = []
    for i in range(b):
        cluster_no = similarities[i][0]
        for doc in range(len(clustered_docs[cluster_no])):  # doc is the index of each document in its cluster
            clustering_results.append(
                [clustered_docs[cluster_no][doc],
                 similarity(given_docs_embedding[clustered_docs[cluster_no][doc]], q_vector)])
    clustering_results = sorted(clustering_results, key=lambda x: (x[1]))
    clustering_results.reverse()
    for i in range(k):
        print(topics[clustering_results[i][0]])
        print(urls[clustering_results[i][0]])
        print(clustering_results[i][1])
        print()

if part == "2":
    # Reading The unlabeled Data
    unlabeled_dataset = pd.read_excel('IR1_7k_news.xlsx')
    unlabeled_docs = unlabeled_dataset['content']
    unlabeled_urls = unlabeled_dataset['url']
    unlabeled_docs = unlabeled_docs[:docs_to_label]
    unlabeled_count = len(unlabeled_docs)
    # Preprocessing

    # Normalizing
    for doc in unlabeled_docs:
        doc = normalizer.normalize(doc)

    # Tokenizing and Stemming
    token_stream = []
    for docid in range(len(unlabeled_docs)):
        terms = word_tokenize(unlabeled_docs[docid])
        for pos in range(len(terms)):
            term = stemmer.stem(terms[pos])
            term = terms[pos]
            token_stream.append([term, [docid, pos]])
    token_stream.sort(key=lambda x: x[0])

    # index construction
    positional_index = [token_stream[0]]

    for token in token_stream:
        if token[0] != positional_index[-1][0]:
            positional_index.append(token)
        elif token[1][0] != positional_index[-1][-1][0]:
            positional_index[-1].append(token[1])
        else:
            positional_index[-1][-1].append(token[1][1])
    # frequency calculation
    term_freq = []
    freqs = []
    for term in positional_index:
        docs = term[1:]
        freq = 0
        for docid in docs:
            freq += len(docid) - 1
        term_freq.append([freq, term[0]])
    term_freq.sort()
    term_freq.reverse()

    # stop word removal
    for i in range(15):
        for term in positional_index:
            if term[0] == term_freq[i][1]:
                positional_index.remove(term)

    # Ranked Retrieval
    # tf idf docs construction
    weight_postings_list = {}
    unlabeled_tf_idf_docs = [{} for sub in range(unlabeled_count)]
    for term in positional_index:
        docs = term[1:]
        df = len(term) - 1
        weight_postings_list[term[0]] = tf_idf(1, df, unlabeled_count)
        for doc in docs:
            tf = len(doc) - 1
            unlabeled_tf_idf_docs[doc[0]][term[0]] = tf_idf(tf, df, unlabeled_count)
    unlabeled_docs_embedding = doc_embed(unlabeled_tf_idf_docs, w2v_model)
    labels = {"sport": [], "economy": [], "political": [], "culture": [], "health": []}
    for unlabeled_doc in range(
            len(unlabeled_docs_embedding)):  # finding distances (unlabeled_doc = doc_id of unlabeled document in 7k set)
        print(unlabeled_doc)
        unlabeled_doc_similarity = {}
        for labeled_doc_id in range(len(given_docs_embedding)):
            unlabeled_doc_similarity[labeled_doc_id] = similarity(given_docs_embedding[labeled_doc_id],
                                                                  unlabeled_docs_embedding[unlabeled_doc])
        d = Counter(unlabeled_doc_similarity)
        KNN = d.most_common(K)
        KNN_topics = {"sport": 0, "economy": 0, "political": 0, "culture": 0, "health": 0}
        for item in KNN:
            KNN_topics[topics[item[0]]] += 1
        this_doc_topic = max(KNN_topics.items(), key=operator.itemgetter(1))[0]
        labels[this_doc_topic].append(unlabeled_doc)

    # query processing
    query = input()
    query = normalizer.normalize(query)
    # query = stemmer.stem(query)
    query = word_tokenize(query)
    cat = query[query.index(":") + 1]
    query.remove("cat")
    query.remove(":")
    query.remove(cat)
    # q vector constrcution
    q_vector = np.zeros(300)

    weight_sum = 0
    for word in query:
        q_vector += w2v_model.wv[word] * weight_postings_list[word]
        weight_sum += weight_postings_list[word]
    q_vector = q_vector / weight_sum

    results = {}
    for doc_id in labels[cat]:
        results[doc_id] = similarity(q_vector, unlabeled_docs_embedding[doc_id])

    d = Counter(results)
    top_k_results = d.most_common(k)
    for result in top_k_results:
        print(unlabeled_urls[result[0]])
