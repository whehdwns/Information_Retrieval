import pandas as pd
import numpy as np
import nltk
import re
import os
import math
import string
import time

from sklearn import feature_extraction
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm


columns =['Assessment', 'Docid', 'Title', 'Authors', 'Journal', 'ISSN', 'Year','Language', 'Abstract','Keywords']

train_df = pd.read_csv('phase1.train.shuf.tsv',sep='\t',header=None, names=columns)
dev_df = pd.read_csv('phase1.dev.shuf.tsv',sep='\t',header=None, names=columns)
test_df = pd.read_csv('phase1.test.shuf.tsv',sep='\t',header=None,names=columns)


train_df.head(2)

train_df.groupby(['Assessment']).count()

len(train_df)


print('# of Positive (-1): '+  str(train_df.groupby(['Assessment']).count()['Docid'][1] /len(train_df)*100) + ' %')
print('# of Negative (1): '+ str(train_df.groupby(['Assessment']).count()['Docid'][-1]/len(train_df)*100) + ' %')


train_df['Language'].unique()

print('Size of train dataset', train_df.shape)
print('Size of dev dataset', dev_df.shape)
print('Size of test dataset', test_df.shape)

# Normalization
#     lower-case words
#     Change short term to long terms for verb.
#     remove punctuation
#         https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
#     remove numbers

def normalization(word):
    word = word.replace("'",'')
    word  = re.sub(r'[^\w\s]', '', word)
    word = word.translate(str.maketrans('', '', string.punctuation))
    return word


# Preprocess dataset 
#   Normalization
#   Removed Stop words 
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


def precision_recall_f1score(pred, truth):
    TP, FP, FN, TN = 0, 0, 0, 0
    for p, t in zip(pred, truth):
        if p==1:
            if t ==1:
                TP +=1
            if t == -1:
                FP +=1
        if p ==-1:
            if t == 1:
                FN+=1
            if t == -1:
                TN +=1
    precision = TP / (TP + FP)    
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall/(precision+recall)
    corr,total = TN+TP , TN+TP+FN+FP
    accuracy = corr/total
    print("Recall: {}/{} = {:.4f} %".format(TP, TP+FN, recall*100))
    print("Precision: {}/{} = {:.4f} %".format(TP, TP+FP, precision*100) )
    print("F1 score: {:.4f}".format(f1_score))
    print("Accuracy: {}/{} = {:.4f} %".format(corr,total,accuracy*100) )
    return recall, precision, f1_score, accuracy

train_clean_df = preprocess(train_df['Title'])
dev_clean_df = preprocess(dev_df['Title'])
test_clean_df =  preprocess(test_df['Title'])

vectorize = TfidfVectorizer()
train_X = vectorize.fit_transform(train_clean_df)
dev_X= vectorize.transform(dev_clean_df)
train_Y = train_df['Assessment']
print(' ')
print('BernoulliNB')
clf_nb = BernoulliNB(alpha=0.1)
clf_nb.fit(train_X, train_Y)

print("Train BernoulliNB Title")
preds_nb = clf_nb.predict(train_X)
precision_recall_f1score(preds_nb, train_df['Assessment'])

print("Dev_X BernoulliNB Title")
preds_nb_dev = clf_nb.predict(dev_X)
precision_recall_f1score(preds_nb_dev, dev_df['Assessment'])

print('')
# ## Experiment #1: Is More Text Better?
print('Experiment #1: Is More Text Better?')
train_df.head(1)
train_df['Abstract'] = train_df['Abstract'].fillna('')
dev_df['Abstract'] = dev_df['Abstract'].fillna('')
test_df['Abstract'] = test_df['Abstract'].fillna('')

train_df['Keywords'] = train_df['Keywords'].fillna('')
dev_df['Keywords'] = dev_df['Keywords'].fillna('')
test_df['Keywords'] = test_df['Keywords'].fillna('')

train_long_df = train_df.assign(More_Text = lambda train_en: train_en['Title'] + " " + train_en['Abstract'] + " " + train_en['Keywords'])
dev_long_df = dev_df.assign(More_Text = lambda dev_en: dev_en['Title'] + " " + dev_en['Abstract'] + " " + dev_en['Keywords'])
test_long_df = test_df.assign(More_Text = lambda test_en: test_en['Title'] + " " + test_en['Abstract'] + " " + test_en['Keywords'])

print("Title, Abstract, Keywords")

train_clean_long_df = preprocess(train_long_df['More_Text'])
dev_clean_long_df = preprocess(dev_long_df['More_Text'])
test_clean_long_df =  preprocess(test_long_df['More_Text'])

vectorize = TfidfVectorizer()
train_long_X = vectorize.fit_transform(train_clean_long_df)
dev_long_X= vectorize.transform(dev_clean_long_df)
train_long_Y = train_long_df['Assessment']


print('BernoulliNB')
clf_nb_long = BernoulliNB(alpha=0.1)
clf_nb_long.fit(train_long_X, train_long_Y)

print("Train_long_X BernoulliNB")
preds_nb_long = clf_nb_long.predict(train_long_X)
precision_recall_f1score(preds_nb_long, train_long_df['Assessment'])

print("Dev_long_X BernoulliNB")
preds_nb_dev_long = clf_nb_long.predict(dev_long_X)
precision_recall_f1score(preds_nb_dev_long, dev_long_df['Assessment'])


# ## Experiment #2: Surprise Me
# Conduct another experiment, preferably something non-trivial. 
# 1. Using Another Machine Learning Model
print('')
print("Experiment #2")
print("Using Using Another Machine Learning Model")
print('')
print("KNN")
clf_knn=KNeighborsClassifier(n_neighbors=5, weights='distance')
clf_knn.fit(train_long_X, train_long_Y)
print("Train_long_X KNN")
preds_knn = clf_knn.predict(train_long_X)
precision_recall_f1score(preds_knn, train_long_df['Assessment'])

print("Dev_long_X KNN")
preds_knn_dev_long =clf_knn.predict(dev_long_X)
precision_recall_f1score(preds_knn_dev_long, dev_long_df['Assessment'])

print('')
# ### PassiveAggressiveClassifier
print("PassiveAggressiveClassifier")

clf_pass = PassiveAggressiveClassifier(max_iter=100)
clf_pass.fit(train_long_X, train_long_Y)

print("Train_long_X PassiveAggressiveClassifier")
preds_pass = clf_pass.predict(train_long_X)
precision_recall_f1score(preds_pass, train_long_df['Assessment'])

print("Dev_long_X PassiveAggressiveClassifier")
preds_pass_dev_long =clf_pass.predict(dev_long_X)
precision_recall_f1score(preds_pass_dev_long, dev_long_df['Assessment'])

print('')
# ### SVM
print("SVM Linear SVC")
clf_svm = svm.LinearSVC(C=2)
clf_svm.fit(train_long_X, train_long_Y)


print("Train_long_X SVM")
preds_svm = clf_svm.predict(train_long_X)
precision_recall_f1score(preds_svm, train_long_df['Assessment'])

print("Dev_long_X SVM")
preds_dev_long =clf_svm.predict(dev_long_X)
precision_recall_f1score(preds_dev_long, dev_long_df['Assessment'])



test_long_X= vectorize.transform(test_clean_long_df)
pred_svm = clf_svm.predict(test_long_X)

with open('dcho13.txt', 'w') as f:
    for x in zip(test_df['Docid'], pred_svm):
        f.write('%s\t%i\n' % x)

