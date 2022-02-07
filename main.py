from __future__ import unicode_literals
import math
import pandas as pd
from hazm import *
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from numpy.linalg import norm
import multiprocessing

k = 5


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
        results.append(similarity(q_vector, document))
    results = np.array(results)
    results = results.argsort()[-k:][::-1]
    return results


cores = multiprocessing.cpu_count()

# Reading The Data
dataset = pd.read_excel('IR1_7k_news.xlsx')
docs = dataset['content']
N = len(docs)
titles = dataset['title']
# docs = docs[:30]
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

# training model
my_w2v_model = Word2Vec(min_count=1, window=5, vector_size=300, alpha=0.03, workers=cores - 1)
# # make training data
training_data = []
for doc in tf_idf_docs:
    training_data.append(doc.keys())

# train
my_w2v_model.build_vocab(training_data)
my_w2v_model.train(training_data, total_examples=my_w2v_model.corpus_count, epochs=20)
# docs
my_model_docs_embedding = doc_embed(tf_idf_docs, my_w2v_model)
w2v_model = Word2Vec.load("w2v_150k_hazm_300_v2.model")  # given model
given_docs_embedding = doc_embed(tf_idf_docs, w2v_model)
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

results_of_given_model = query_similarity_with_all(my_model_docs_embedding, q_vector)
results_of_my_model = query_similarity_with_all(given_docs_embedding, q_vector)
print("given model results")
for i in range(k):
    print(titles[results_of_given_model[i]])
    print(results_of_given_model[i])

print("my model results")
for i in range(k):
    print(titles[results_of_my_model[i]])
    print(results_of_my_model[i])

