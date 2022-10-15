#!/usr/bin/env python
# coding: utf-8

import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter, defaultdict, OrderedDict
import re
import sys
import os
import math
import string
import time
import datetime
import operator

execution_start = time.time()

cord19_file=open('data/cord19/cord19.txt',"r", encoding="UTF-8")
cord19_content = cord19_file.read()

cord19_key_file =open('data/cord19.topics.keyword.txt',"r")
cord19_key_content = cord19_key_file.read()

cord19_qs_file =open('data/cord19.topics.question.txt',"r")
cord19_qs_content = cord19_qs_file.read()

regextoken_P = RegexpTokenizer(r'<P ID=\d+>(.*?)</P>')
regextoken_Q = RegexpTokenizer(r'<Q ID=\d+>(.*?)</Q>')

cord19_text_list = regextoken_P.tokenize(cord19_content)
cord19_key_text_list = regextoken_Q.tokenize(cord19_key_content)
cord19_qs_text_list = regextoken_Q.tokenize(cord19_qs_content)

# Normalization
#     lower-case words
#     Change short term to long terms for verb.
#     remove punctuation
#         https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
#     remove numbers
def normalization(word):
    word= word.lower()
    word = word.replace("'re",' are').replace("'m'", ' am').replace("'s",' is').replace("n't",' not').replace("'ve",' have')\
    .replace("'d",' had').replace("'ll",' will')
    word = word.replace("'",'')
    word  = re.sub(r'[^\w\s]', '', word)
    word = word.translate(str.maketrans('', '', string.punctuation))
    word = re.sub('[0-9]', '', word)
    return word


# Calculate Terms
# 1. Normalized the text 
# 2. It tokenized the text and count the occurence of the text
# 3. It returns the document id and count for each of the terms.
# 4. It also count the number of terms and documents. 
def calculate_terms(listed):
    normalized_text = []
    collection_frequency = Counter()
    document_frequency = Counter()
    output_wordlist_dict ={}
    terms_frequency = defaultdict(lambda: Counter([]))
    
    for i in range(len(listed)):
        normalized_text.append(normalization(listed[i]))
    
    for i in range(len(normalized_text)):
        tokenized_list= []
        for j in normalized_text[i].split():
            tokenized_list.append(j)
        output_wordlist_dict[i] = Counter(tokenized_list)
        collection_frequency.update(tokenized_list)
        document_frequency.update(set(tokenized_list))
        
    for key, value in output_wordlist_dict.items():
        for term, term_cnt in value.items():
            terms_frequency[term][key] += term_cnt
    
    return normalized_text, collection_frequency, document_frequency, output_wordlist_dict, terms_frequency

# Dictionary List
# 1. It stores information about a term (Term Frequency and offset)
# 2. The dictionary is sorted by term. 
# 3. It starts from 0 and counts the number document frequency and term count. 
def dictionary_list(listed):
    sort_dict = {}
    result_sort_dict = {}
    offset_sum = 0
    offset_i = 0
    sort_dict = OrderedDict(sorted(listed.items()))
    for i, value in enumerate(sort_dict.keys()):
        offset_i = len(sort_dict[value]) * 2 
        result_sort_dict[value] = len(sort_dict[value].values()),offset_sum
        offset_sum = offset_sum + offset_i 
    return result_sort_dict

# Inverted file
# 1. It stores the sorted entries as an inverted file
def inverted_file(key, dict_listed):
    inverted_list = []
    for i in key:
        for docid, term_cnt in dict_listed[i].items():
            inverted_list.append(docid)
            inverted_list.append(term_cnt)
    return inverted_list

# Store_Inverted_bin
# 1. It stores the inverted file as binary file. 
# 2. It stored this binary file as 4-byte integers. 
def Store_Inverted_bin(file, name):
    with open("Inverted_File/inverted_file_"+name+"_binary.bin", "wb") as fb:
        for num in file:
            fb.write(num.to_bytes(4, "big"))
    print("Inverted File " + name +" is created.")

# Calculate IDF
# 1. It gets posting list and length of the document
# 2. In posting list, it contains (frequency of the terms, offset of the terms)
# 3. To calculate the IDF, log2(Number of document / document frequency) 
def idf_corpus(dict_corpus,N_corpus):
    idf_dict = {}
    for key_i in dict_corpus.keys():
        tf_i = dict_corpus.get(key_i)[0]
        idf_i = math.log2(N_corpus/tf_i)
        idf_dict[key_i] = idf_i
    return idf_dict

#DFx IDF
# It gets term frequency for each documents.
# 1. It iterates through documents.
# 2. It iterates through terms in document.
# 3. If the term in document does not exit in IDF, it sets to 0.
# 4. Else it multiplies term freqeuncy by IDF.

def tf_idf(post_list,idf_matrix):
    weight_matrix =[]
    for i, j in post_list.items():
        idf ={}
        for k in j:
            if k not in idf_matrix:
                idf_matrix[k] = 0
            else:
                idf[k] = idf_matrix[k]*j[k]
        weight_matrix.append(idf)                       
    return weight_matrix

#vector length
# It computes the vector lengths.
# 1. It iterates through document and query.
# 2. It takes the TF-IDF for each document and query, and square it.
# 3. Then, it gets sum of squares by adding the square of TF-IDF.
# 3. It takes square root of the sum of squares.
def vector_length(weight):
    length_matrix = {}
    for doc_i in range(len(weight)):
        length_matrx = []
        for i in weight[doc_i].values():
            length_matrx.append(i)
        sum_of_squares = sum(map(lambda k : k * k, length_matrx))
        vlength = math.sqrt(sum_of_squares)
        length_matrix[doc_i] = vlength
    return length_matrix

# Cosine Similarities
# 1. It loops through query.
# 2. It loops through each terms in each query.
# 3. It gets tf-idf weight of term in query.
# 4. It loops through document and computes the cosine similarities.
# 5. To avoid dividing by zero, it checks whether the tf-idf weight of term in document is zero or not. 
# 6. Cosine similarities = (Document Length * Query Length)/(tf-idf weight of term in document*tf-idf weight of term in query)
def cosine_similarities(doc_weight, query_weight, doc_length, query_length, query_term_freq):
    N = len(doc_weight)
    cos_score = []
    for i in range(len(query_term_freq)):
        cos_score.append([0]*N)
        for j  in query_term_freq[i].keys():
            query_tfidf = 0
            if query_weight[i].get(j):
                query_tfidf = query_weight[i].get(j)
            for k in range(len(doc_weight)):
                if(query_length[i] != 0) & (doc_length[k] != 0):
                    if(doc_weight[k].get(j)):
                        #Document Length * Query Length
                        doc_query_length = doc_length[k] * query_length[i]
                        # tf-idf weight of term in document * tf-idf weight of term in query
                        doc_query_vector = doc_weight[k].get(j) * query_tfidf
                        cos_score[i][k] += doc_query_vector / doc_query_length  
    return cos_score

# Score Ranking
# 1. It iterates through cosine scores for each query.
# 2. It iterate through each document in query to gets document id and score.
# 3. It sorts the cosine score in the descending order.
# 4. It takes top 100 cosine score. 
# 5. It creates text file (Query ID, Q0, Cosine Score, Ranks, Document ID, JHUID)
def score_ranking(doc_weight, cos_score, jhuid, filename):
    print("Creating Score Ranking")
    N = len(doc_weight)
    score_results = []
    for score in cos_score:
        result =[]
        for i in range(N):
            result.append((score[i], i))
        result.sort(reverse= True)
        score_results.append(result)
    score_output = open(filename, "w")
    for query_id in range(len(score_results)):
        for j in range(100):
            doc_id, cos_score = score_results[query_id][j]
            score_output.write(str(query_id+1) + " Q0 " + str(cos_score) + " " +  str(j+1) + " " + str(doc_id) + " " + jhuid + '\n')
    score_output.close()
    print("Score Ranking file (" + filename+ " ) is created")
    

cord19_normalized_text, cord19_collection_freq, cord19_document_freq, cord19_term_freq, cord19_posting_list_output = calculate_terms(cord19_text_list)
print("Cord19")
print('Number of paragraph:', len(cord19_normalized_text))
print('Number of unique words observed:', len(cord19_document_freq))
print('The total number of words encountered:', sum(cord19_collection_freq.values()))

cord19_key_normalized_text, cord19_key_collection_freq, \
cord19_key_document_freq, cord19_key_term_freq, cord19_key_posting_list_output = calculate_terms(cord19_key_text_list)
print("Cord19 Keyword")
print('Number of paragraph:', len(cord19_key_normalized_text))
print('Number of unique words observed:', len(cord19_key_document_freq))
print('The total number of words encountered:', sum(cord19_key_collection_freq.values()))

cord19_qs_normalized_text, cord19_qs_collection_freq, \
cord19_qs_document_freq, cord19_qs_term_freq, cord19_qs_posting_list_output = calculate_terms(cord19_qs_text_list)
print("Cord19 Question")
print('Number of paragraph:', len(cord19_qs_normalized_text))
print('Number of unique words observed:', len(cord19_qs_document_freq))
print('The total number of words encountered:', sum(cord19_qs_collection_freq.values()))

#Cord19 
cord19_index_start = time.time()
cord19_dict_pos_output = dictionary_list(cord19_posting_list_output)
cord19_byte_file = inverted_file(cord19_dict_pos_output.keys(), cord19_posting_list_output)
Store_Inverted_bin(cord19_byte_file, "cord19")
original_sized = os.path.getsize('data/cord19/cord19.txt')
print('Size of Cord19 original_text: ' + str(round(original_sized/(1024*1024), 4)) + ' MB')
Inverted_size = os.path.getsize('Inverted_File/inverted_file_cord19_binary.bin')
print('Size of Cord19 Inverted File: ' + str(round(Inverted_size/(1024*1024), 4)) + ' MB')
cord19_index_end = time.time()
cord19_index_time = str(datetime.timedelta(seconds=cord19_index_end-cord19_index_start)).split(".")
print('Execution time of building Cord19 dictionary and inverted index: ' + cord19_index_time[0])
print("")


print("")
process_start = time.time()
print("Computing IDF, TF-IDF, Vector Length for Cord19 Document")
idf_matrix = idf_corpus(cord19_dict_pos_output,len(cord19_normalized_text))
cord19_weight = tf_idf(cord19_term_freq, idf_matrix)
cord19_length = vector_length(cord19_weight)

print("Computing IDF, TF-IDF, Vector Length for Cord19 Keyword Query")
cord19_key_weight = tf_idf(cord19_key_term_freq, idf_matrix)
cord19_key_length = vector_length(cord19_key_weight)

print("Computing IDF, TF-IDF, Vector Length for Cord19 Question Query")
cord19_qs_weight = tf_idf(cord19_qs_term_freq, idf_matrix)
cord19_qs_length = vector_length(cord19_qs_weight)

print("Computing Cosine Score for Keyword")
cos_score_keyword = cosine_similarities(cord19_weight, cord19_key_weight, cord19_length, cord19_key_length, cord19_key_term_freq)
score_ranking(cord19_weight, cos_score_keyword, 'dcho13','Result_output/dcho13-a.txt' )

print("Computing Cosine Score for Question")
cos_score_question = cosine_similarities(cord19_weight, cord19_qs_weight, cord19_length, cord19_qs_length, cord19_qs_term_freq)
score_ranking(cord19_weight, cos_score_question, 'dcho13','Result_output/dcho13-b.txt' )
process_end = time.time()
process_time = str(datetime.timedelta(seconds=process_end-process_start)).split(".")
print('Execution time of process all topics: ' + process_time[0])

print("")
execution_end = time.time()
exeuction_time = str(datetime.timedelta(seconds=execution_end-execution_start)).split(".")
print('Total Execution Time: ' + exeuction_time[0])
print("")
