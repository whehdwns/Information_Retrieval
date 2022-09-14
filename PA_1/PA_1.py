# Programming Assignment #1
# Dongjun(DJ) Cho
# EN.605.744 Information Retrieval

import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import re
import sys

# Normalization
#   convert text to lower-case words
#   change short verb term to long verb terms.
#   remove punctuation
def normalization(word):
    word= word.lower()
    word = word.replace("'m'", ' am').replace("'re",' are').replace("'s",' is')
    word = word.replace("n't",' not').replace("'ve",' have').replace("'d",' had').replace("'ll",' will')
    word = word.replace("'",'')
    word  = re.sub(r'[^\w\s]', '', word)
    return word

# Calculate_word
#   Normalize the text
#   Split the text and Count the number of terms in the text
#   Updates the collection frequency and document frequency if there is new terms and document.
#   Uses Dictionary to store collection frequency and document frequency. 
def calculate_word(listed):
    normalized_text = []
    collection_frequency = Counter()
    document_frequency = Counter()
    combined_dictionary ={}
    
    for i in range(len(listed)):
        normalized_text.append(normalization(listed[i]))

    for i in range(len(normalized_text)):
        tokenized_list= []
        for j in normalized_text[i].split():
            tokenized_list.append(j)
        collection_frequency.update(tokenized_list)
        document_frequency.update(set(tokenized_list))
        
    for i in collection_frequency:
        combined_dictionary[i] = [collection_frequency[i], document_frequency[i]]
    return normalized_text, collection_frequency, document_frequency , combined_dictionary

# dictionary_terms_occurence
#   Counts the number of terms that occurs in certain number. 
def dictionary_terms_occurence(doc_freq, num):
    dictionary_terms =[]
    for key, value in doc_freq.items():
        if value == num:
            dictionary_terms.append(key)
    return dictionary_terms

# main
#   Read text file
#   Uses RegexpTokenizer (NLTK) to extract only the text from text file.
#   calculate_word() returns the number of paragraphs, collection frequency, document frequency, combined dictionary
#   Sorts the stored dictionary based on collection frequency
#   Counts the number of terms that occurs in one documents
#   print number of paragraph, unique words, total number of words, 
#      Top 100 most frequent words, 500th, 1000th, 5000th ranked words, number of words that occurs in one document,
#      % of dictionary terms occur in one document (number of words that occurs in one document / number of unique words(size of document frequency)
def main():
    with open(sys.argv[1], 'r') as f:
        contents = f.read()

    regextoken = RegexpTokenizer(r'<P ID=\d+>(.*?)</P>')

    text_list = regextoken.tokenize(contents)

    normalized_list, collection_freq, document_freq, combined_dict = calculate_word(text_list)
    sorted_list_by_freq = sorted(combined_dict.items(), key=lambda r: r[1][0], reverse=True)
    dict_occurence = dictionary_terms_occurence(document_freq, 1)

    print('Number of paragraph:', len(normalized_list))
    print('Number of unique words observed:', len(combined_dict))
    print('The total number of words encountered:', sum(collection_freq.values()))

    print('The 100 most frequent words: ', sorted_list_by_freq[:100])

    print('500th ranked words: ',sorted_list_by_freq[499])
    print('1000th ranked words: ',sorted_list_by_freq[999])
    print('5000th ranked words: ',sorted_list_by_freq[4999])

    print('The number of words that occur in exactly one document:', len(dict_occurence))
    print('The percentage of the dictionary terms which occur in just one document (%):', round((len(dict_occurence)/len(document_freq))*100,2))
    
main()

