from __future__ import unicode_literals
import math
import pandas as pd
from hazm import *
import numpy as np
import matplotlib.pyplot as plt

k = 100


def tf_idf(term_frequency, doc_frequency, N):
    tf = 1 + math.log10(term_frequency)
    idf = math.log10(N / doc_frequency)
    tf_idf_weight = tf * idf

    return tf_idf_weight


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
# weight posting list and champion list construction
weight_postings_list = []
champion_list = []
for term in positional_index:
    weight_postings_list.append([term[0]])
    champion_list.append([term[0]])
    docs = term[1:]
    df = len(term) - 1
    for doc in docs:
        docid = doc[0]
        tf = len(doc) - 1
        weight = tf_idf(tf, df, N)

        weight_postings_list[-1].append([docid, weight])
    tmp = weight_postings_list[-1][1:]
    tmp = sorted(tmp, key=lambda l: l[1], reverse=True)
    for i in range(k):
        try:
            champion_list[-1].append(tmp[i])
        except IndexError:
            break
    champion_list[-1].append(df)
# query processing
query = input()
query = normalizer.normalize(query)
# query = stemmer.stem(query)
query = word_tokenize(query)
# q vector constrcution
q_vector = []
q_vector.append(0)
documents_score = []
for word in query:
    # for term in weight_postings_list:   # no champion list
    #     if term[0] == word:
    #         df = len(term) - 1
    #         break
    for term in champion_list:  # with champion list
        if term[0] == word:
            df = term[-1]
            break
    weight = tf_idf(1, df, N)
    q_vector.append([word, weight])
    tmp = term[1:]
    del tmp[-1:]
    documents_score.append(tmp)

documents_vectors = []

for i in range(len(query)):
    for doc in documents_score[i]:  # each doc of the 'i'th word , each doc is like [714, 2.1077846613843727]
        flag = True
        for vec in documents_vectors:  # check if this doc already has a vector ,
            # each vec is like [docid,[word,weight],[word,weight],...]
            if vec[0] == doc[0]:
                if len(vec) == i + 1:  # if vector had all the word until this word
                    # vec.append([query[i], doc[1] * q_vector[i + 1][1]])
                    vec.append([query[i], doc[1]])
                else:  # insert 0 weights for missing words
                    for missing in range(i + 1 - len(vec)):
                        vec.append(["temporarykossher", 0])

                    for v in range(1, len(vec)):
                        if vec[v][0] == "temporarykossher":
                            vec[v][0] = query[v - 1]
                    vec.append([query[i], doc[1] * q_vector[i + 1][1]])  # now enter the last

                flag = False
                break
        if flag:  # vector construction
            documents_vectors.append([doc[0]])
            for j in range(i):
                documents_vectors[-1].append([query[j], 0])
            documents_vectors[-1].append([query[i], doc[1]])

for vect in documents_vectors:
    if len(vect) < len(query) + 1:  # add 0 weights to vector without last words
        for missing in range(len(query) + 1 - len(vect)):
            vect.append(["temporarykossher", 0])
        for v in range(1, len(vect)):
            if vect[v][0] == "temporarykossher":
                vect[v][0] = query[v - 1]

doc_scores = []
for vect in documents_vectors:  # final score calculator
    doc_scores.append([vect[0], 0])
    vec_size = 0
    for i in range(1, len(vect)):
        doc_scores[-1][1] += vect[i][1] * q_vector[i][1]
        vec_size += vect[i][1] ** 2
    vec_size = vec_size ** 0.5
    doc_scores[-1][1] = doc_scores[-1][1] / vec_size

doc_scores = sorted(doc_scores, key=lambda l: l[1], reverse=True)


for i in range(len(doc_scores)):
    print(titles[doc_scores[i][0]])
    print([doc_scores[i][0],doc_scores[i][1]])


