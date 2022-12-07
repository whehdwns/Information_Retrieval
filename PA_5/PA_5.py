import pandas as pd
import numpy as np
import nltk
import re
import os
import math
import string
import time
import sys
import glob
import random
import hashlib as hl
import tracemalloc

nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")

path = os.getcwd()
testfilesd = glob.glob(path + "/dataset/duplicatetests/*.tsv")
testfilesk = os.listdir('dataset/twoktests/')

thirty_df_orig = pd.read_csv(testfilesd[4],sep='\t+|\n',header=None, names=['Context'])
hundred_df_orig = pd.read_csv(testfilesd[0],sep='\t+|\n',header=None, names=['Context'])
threehundred_df_orig = pd.read_csv(testfilesd[6],sep='\t+|\n',header=None, names=['Context'])
onek_df_orig = pd.read_csv(testfilesd[2],sep='\t+|\n',header=None, names=['Context'])
threek_df_orig = pd.read_csv(testfilesd[7],sep='\t+|\n',header=None, names=['Context'])
tenk_df_orig = pd.read_csv(testfilesd[3],sep='\t+|\n',header=None, names=['Context'])
thirtyk_df_orig = pd.read_csv(testfilesd[5],sep='\t+|\n',header=None, names=['Context'])
hundredk_df_orig = pd.read_csv(testfilesd[1],sep='\t+|\n',header=None, names=['Context'])

thirty_df = thirty_df_orig.copy()
hundred_df = hundred_df_orig.copy()
threehundred_df = threehundred_df_orig.copy()
onek_df = onek_df_orig.copy()
threek_df = threek_df_orig.copy()
tenk_df = tenk_df_orig.copy()
thirtyk_df = thirtyk_df_orig.copy()
hundredk_df = hundredk_df_orig.copy()


# Normalization
#     lower-case words
#     Change short term to long terms for verb.
#     remove punctuation
#         https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
def normalization(word):
    word = word.replace("'",' ')
    word = word.replace("'re",' are').replace("'m'", ' am').replace("'s",' is').replace("n't",' not')
    word = word.replace("'ve",' have').replace("'d",' had').replace("'ll",' will')
    word  = re.sub(r'[^\w\s]', ' ', word)
    word = word.translate(str.maketrans('', '', string.punctuation))
    return word

# Preprocess dataset 
#   Normalization
#   Removed English Stop words 
def preprocess(data):
    result = []
    for line in data:
        word = normalization(line)
        word = word.lower().strip().split()
        stopwords = nltk.corpus.stopwords.words("english")
        word = [w for w in word if not w in stopwords]
        word = " ".join(word)
        result.append(word)
    return result


# N_gram/ K-shingles
#   Represent the document as a n-gram
#   It splits the document by N words.
def N_gram(text, N):
    grams_list=set()
    text = text.split()
    for i in range(len(text)-N+1):
        shingle = text[i:i+N]
        shingle = ' '.join(shingle)
        grams_list.add(shingle)
    return grams_list


# Hashing
#   It represents strings into 64 bit integers using hashing
def hashing(text):
    return int.from_bytes(hl.sha256(text.encode("utf-8")).digest()[:8], 'little') # 64-bit int


# random_hash_func
#       Random Hash Function for Creating signature matrix.
def random_hash_func(N):
    a = random.randint(N+1,(2**64 - N))
    b = random.randint(N,(2**64 - N))
    return lambda x: (a * x + b) % N

# Make Hashes
#   It stores each hash function for the document. 
def hashe_func(N, num_hashes):
    list_of_hash_fn=[]
    for i in range(num_hashes):
        list_of_hash_fn.append(random_hash_func(N))
    return list_of_hash_fn 


# Shingledhash
#   Using N-gram, it splits the string into N-gram
#   Then, it represetns the N-gram into integer using hashing. 
def shingledhash(df, N):
    docsAsShingleSets = {}
    for i in range(1,len(df)+1):
        n_gram_list=N_gram(df[i], N)
        shinglesInDoc = set()
        for j in range(len(n_gram_list)):
            shinglesInDoc.add(hashing(list(n_gram_list)[j]))
        docsAsShingleSets[i] = shinglesInDoc
    return docsAsShingleSets

# shingles_doc
#  It combines all shingles in the document.
#  It stores each shingles appear in the document. 
#  Then, it sorted the combined shingles. 
def shingles_doc(shingled_documents):
    list_of_tuples = []
    list_of_documentid =[]
    for i in shingled_documents:
        list_of_documentid.append(i)
        for j in shingled_documents[i]:
            list_of_tuples.append((j, i))
    list_of_tuples.sort()
    return list_of_tuples, list_of_documentid


# Minhash signature matrix
#  it creates the minhash signature matirx.
#  it first initalizes the signature matrix with infinity.
#  It generates the random hash function. 
#  Then, it find the corresponding set through the set of hash functions in the signature matrix and take the minimum value.
def make_minhash_signature(shingled_data, num_hash):
    inv_index, docids = shingles_doc(shingled_data)
    sigmatrix = np.full([num_hash, len(docids)], np.inf)  
    hash_funcs = hashe_func(len(inv_index), num_hash)
    for row, docid in inv_index:
        for row1 in range(num_hash):
            sigmatrix[row1,docids.index(docid)]=min(sigmatrix[row1,docids.index(docid)],hash_funcs[row1](row))
    return sigmatrix

# Jacard similarity
# It uses jacard similiarty to compute similarity
# It divides intersection of two document by union of two documents. 
def jacard_similarity(id1, id2, minhash_sigmat, docids):
    return np.mean(minhash_sigmat[:, docids.index(id1)]==minhash_sigmat[:, docids.index(id2)])

# Near Document
#  It iterates through documents and compute jacard similairty for document pairs.
#  If the jacard similiarty is greater than equal to threshold, it knows the document is near duplicate.
def near_document(df, sigmin, threshold):
    near_docs=[]
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            minhash_similar=jacard_similarity(list(df.keys())[i], list(df.keys())[j], sigmin, list(df.keys()))
            minhash_tuples = [list(df.keys())[i], list(df.keys())[j],minhash_similar]
            if minhash_similar>=threshold:
                near_docs+=[minhash_tuples]
    return near_docs


# This function is to handle duplicate document number in cluster documents.
def cluster_doc(df, N):
    comb_doc=[]
    for i in range(1,N+1):
        near =[]
        for j in df:
            if j[0] == i:
                near.append(j[0])
                near.append(j[1])
            else:
                continue
        comb_doc_list =[]
        for k in df:
            for g in near:
                if k[0]==g:
                    comb_doc_list.append(k[0])
                    comb_doc_list.append(k[1])
        comb_doc.append(comb_doc_list)
    sorted_doc =[]
    for i in comb_doc:
        if len(i) !=0:
            sorted_doc.append(sorted(set(i)))
    first_doc=[]
    for i in sorted_doc:
        for j in i[1:]:
            first_doc.append(j)
    merge_doc =[]
    for i in sorted_doc:
        for j in first_doc:
            if i[0] ==j:
                merge_doc.append(i)
    near_ducplicate_list =[]
    for i in sorted_doc:
        if i not in merge_doc:
            near_ducplicate_list.append(i)
    return near_ducplicate_list

# Using cluster document, it creates cluster files. 
def near_document_cluster(near_doc, N):
    doc_id =[]
    for i in near_doc:
        for j in i:
            doc_id.append(j)  
    with open('output/dcho13-'+str(N)+'.txt', 'w') as f:
        for k in range(1, N+1):
            near =[]
            for i in near_doc:
                if i[0] == k:
                    near.append(i)
            if len(near) !=0:
                print(*near[0], file=f)
            if k in doc_id:
                continue
            else:
                print(k, file=f)
    print('output/dcho13-'+str(N)+'.txt is created')



thirty_df['processed_context'] = preprocess(thirty_df['Context'])
shingled_list_30= shingledhash(thirty_df['processed_context'], 3)
minhash_sigmat_30=make_minhash_signature(shingled_list_30, 200)
near_30 = near_document(shingled_list_30, minhash_sigmat_30,0.35)
output_30 = cluster_doc(near_30, 30)
near_document_cluster(output_30,30)


hundred_df['processed_context'] = preprocess(hundred_df['Context'])
shingled_list_100= shingledhash(hundred_df['processed_context'], 3)
minhash_sigmat_100=make_minhash_signature(shingled_list_100, 200)
near_100 = near_document(shingled_list_100, minhash_sigmat_100, 0.35)
output_100 = cluster_doc(near_100, 100)
near_document_cluster(output_100,100)


threehundred_df['processed_context'] = preprocess(threehundred_df['Context'])
shingled_list_300= shingledhash(threehundred_df['processed_context'], 3)
minhash_sigmat_300=make_minhash_signature(shingled_list_300, 200)
near_300 = near_document(shingled_list_300, minhash_sigmat_300, 0.35)
output_300 = cluster_doc(near_300, 300)
near_document_cluster(output_300,300)


onek_df['processed_context'] = preprocess(onek_df['Context'])
shingled_list_1000= shingledhash(onek_df['processed_context'], 3)
minhash_sigmat_1000=make_minhash_signature(shingled_list_1000, 200)
near_1000 = near_document(shingled_list_1000, minhash_sigmat_1000, 0.35)
output_1000 = cluster_doc(near_1000, 1000)
near_document_cluster(output_1000,1000)


path = os.getcwd()
testfilesd2= glob.glob(path + "/dataset/twoktests/*.tsv")
twok_df_orig = pd.read_csv(testfilesd2[0],sep='\t+|\n',header=None, names=['Context'])
twok_df = twok_df_orig.copy()

twok_df['processed_context'] = preprocess(twok_df['Context'])
shingled_list_2000= shingledhash(twok_df['processed_context'], 3)
minhash_sigmat_2000=make_minhash_signature(shingled_list_2000, 100)
near_2000 = near_document(shingled_list_2000, minhash_sigmat_2000, 0.35)
output_2000 = cluster_doc(near_2000, 2000)
near_document_cluster(output_2000,2000)


threek_df['processed_context'] = preprocess(threek_df['Context'])
shingled_list_3000= shingledhash(threek_df['processed_context'], 3)
minhash_sigmat_3000=make_minhash_signature(shingled_list_3000, 200)
near_3000 = near_document(shingled_list_3000, minhash_sigmat_3000, 0.35)
output_3000 = cluster_doc(near_3000, 3000)
near_document_cluster(output_3000,3000)


tenk_df['processed_context'] = preprocess(tenk_df['Context'])
shingled_list_10000= shingledhash(tenk_df['processed_context'], 3)
minhash_sigmat_10000=make_minhash_signature(shingled_list_10000, 200)
near_10000 = near_document(shingled_list_10000, minhash_sigmat_10000, 0.35)
output_10000 = cluster_doc(near_10000, 10000)
near_document_cluster(output_10000,10000)


thirtyk_df['processed_context'] = preprocess(thirtyk_df['Context'])
shingled_list_30000 = shingledhash(thirtyk_df['processed_context'], 3)
minhash_sigmat_30000=make_minhash_signature(shingled_list_30000, 200)
near_30000 = near_document(shingled_list_30000, minhash_sigmat_30000, 0.35)
output_30000 = cluster_doc(near_30000, 30000)
near_document_cluster(output_30000,30000)


hundredk_df['processed_context'] = preprocess(hundredk_df['Context'])
shingled_list_100000 = shingledhash(hundredk_df['processed_context'], 3)
minhash_sigmat_100000=make_minhash_signature(shingled_list_100000, 200)
near_100000 = near_document(shingled_list_100000, minhash_sigmat_100000, 0.35)
output_100000 = cluster_doc(near_100000, 100000)
near_document_cluster(output_100000,100000)

threehundredk_df['processed_context'] = preprocess(threehundredk_df['Context'])
shingled_list_300000 = shingledhash(threehundredk_df['processed_context'], 3)
minhash_sigmat_300000=make_minhash_signature(shingled_list_300000, 200)
near_300000 = near_document(shingled_list_300000, minhash_sigmat_300000, 0.35)
output_300000 = cluster_doc(near_300000, 300000)
near_document_cluster(output_300000,300000)
