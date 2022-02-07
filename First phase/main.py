from __future__ import unicode_literals
import pandas as pd
from hazm import *

# Reading The Data
dataset = pd.read_excel('IR1_7k_news.xlsx')
docs = dataset['content']
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

# Boolean Information Retrieval
query = input()
query = normalizer.normalize(query)
# query = stemmer.stem(query)
query = word_tokenize(query)
if len(query) == 1:  # single word
    results = []
    for term in positional_index:
        if term[0] == query[0]:
            break

    if len(term) < 11:
        for i in range(1,len(term)):
            print(titles[term[i][0]])
    else:
        for i in range(1, 11):
            print(titles[term[i][0]])

else:
    results = []
    queries = []
    for i in range(len(query)):
        for j in range(len(query)):
            queries.append(query[j:i + j + 1])
    queries = set(tuple(element) for element in queries)
    queries = list(list(element) for element in queries)
    queries = sorted(queries, key=len)
    queries.reverse()

    for queri in queries:
        postings_of_query = []
        for i in range(len(queri)):
            for term in positional_index:
                if term[0] == queri[i]:
                    postings_of_query.append(term[1:])

        intersect_list = postings_of_query[0]
        for i in range(len(postings_of_query) - 1):
            temp = []
            for doc in intersect_list:
                for j in range(len(postings_of_query[i + 1])):
                    if doc[0] == postings_of_query[i + 1][j][0]:  # docid 1 = docid 2
                        positions0 = [x + 1 for x in doc[1:]]
                        positions1 = postings_of_query[i + 1][j][1:]
                        positions = list(set(positions0) & set(positions1))
                        if len(positions) != 0:
                            positions.insert(0, doc[0])
                            temp.append(positions)
            intersect_list = temp
        results = results + intersect_list

    ttmp = []
    for i in range(len(results)):
        ttmp.append(results[i][0])
    seen = set()
    seen_add = seen.add
    results =[x for x in ttmp if not (x in seen or seen_add(x))]


    if len(results) < 10:
        for i in range(len(results)):
            print(titles[results[i]])
    else:
        for i in range(10):
            print(titles[results[i]])
