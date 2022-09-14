import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter, defaultdict, OrderedDict
import re
import sys
import os
import time

start_time = time.time()
# with open(sys.argv[1], 'r') as f:
#     contents = f.read()
headline_file=open('headlines.txt',"r")
headline_content = headline_file.read()

regextoken = RegexpTokenizer(r'<P ID=\d+>(.*?)</P>')
headline_text_list = regextoken.tokenize(headline_content)

# Normalization
#     lower-case words
#     Change short term to long terms for verb.
#     remove punctuation
#         https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
def normalization(word):
    word= word.lower()
    word = word.replace("'re",' are').replace("'m'", ' am').replace("'s",' is').replace("n't",' not').replace("'ve",' have')\
    .replace("'d",' had').replace("'ll",' will')
    word = word.replace("'",'')
    word  = re.sub(r'[^\w\s]', '', word)
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
    
    return normalized_text, collection_frequency, document_frequency, terms_frequency

# Dictionary List
# 1. It stores information about a term (Document Frequency and offset)
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
def Store_Inverted_bin(file):
    with open("inverted_fiile_binary.bin", "wb") as fb:
        for num in file:
            fb.write(num.to_bytes(4, "big"))
    print("Inverted File is created.")


normalized_text, collection_freq, document_freq, posting_list_output = calculate_terms(headline_text_list)

postings_list_terms =['heidelberg', 'cesium', 'trondheim', 'crustacean']

for i in postings_list_terms:
    posting_list_output[i]

dict_pos_output = dictionary_list(posting_list_output)
byte_file = inverted_file(dict_pos_output.keys(), posting_list_output)
Store_Inverted_bin(byte_file)


print('Number of paragraph:', len(normalized_text))
print('Number of unique words observed:', len(document_freq))
print('The total number of words encountered:', sum(collection_freq.values()))
print("\n")

print('Size of original_text: ' + str(os.path.getsize('headlines.txt')) + ' bytes')
print('Size of Inverted File: ' + str(os.path.getsize('inverted_fiile_binary.bin')) + ' bytes')
print('Size of Dictionary: ' + str(sys.getsizeof(dict_pos_output)) + ' bytes')
print('Size of Inverted File + Dictionary: '+ str(os.path.getsize('inverted_fiile_binary.bin')+sys.getsizeof(dict_pos_output)) + ' bytes')
print("\n")

# Document Frequency
# 1. It prints the  document frequency and postings list for terms.
# 2. It prints the index of posting list for terms.
# 3. It reads inverted_file binary file every 4 byte, then gets the posting list for the terms. 
def document_freqency(dictionary_output,binary_file, terms):
    term = terms.lower()
    print("document frequency for "+terms+" : ",dictionary_output[term][0])
    if (dictionary_output[term][0] == 0):
        print("The "+ terms + " does not exit in the document.")
    terms_position = dictionary_output[term][1]
    range_of_terms_in_positing_list = list(dictionary_output.keys()).index(term)
    if len(list(dictionary_output)) != range_of_terms_in_positing_list+1:
        next_term = list(dictionary_output.keys())[range_of_terms_in_positing_list+1]
    else:
        print("The terms "+ terms+" is in end of the word. There is no next word")
        return
    next_term_position = dictionary_output[next_term][1]
    range_index = [terms_position, next_term_position]
    print("index of range for "+terms+" : ", range_index)
    list_num = []
    with open(binary_file, "br") as bf:
        for _ in range(terms_position):
            data = bf.read(4)
        for _ in range(terms_position, next_term_position):
            data = bf.read(4)
            number = int.from_bytes(data,"big")
            list_num.append(number)
    print("posting list for "+terms+" : ")
    print(list_num)
    return list_num

postings_list_terms = ['Heidelberg', 'cesium', 'Trondheim', 'crustacean']
for i in postings_list_terms:
    document_freqency(dict_pos_output, "inverted_fiile_binary.bin", i)
    print("-----------------")
print("\n")

word_list = ['Hopkins', 'Stanford', 'Brown', 'college']
for i in word_list:
    print('Document Frequency for '+i+' is '+ str(dict_pos_output[i.lower()]))
print("\n")

elon_list = document_freqency(dict_pos_output, "inverted_fiile_binary.bin", 'Elon'.lower())
musk_list = document_freqency(dict_pos_output, "inverted_fiile_binary.bin", 'Musk'.lower())
print("\n")

set_elon =set(elon_list)
set_musk = set(musk_list)
setone = {1}
set_same = set_elon.intersection(set_musk) - setone
print('Intersection of postings list for each term: Elon and Musk: '+ str(len(list(set_same))))
print(sorted(set_same))
print("\n")

print("Total execution time: %s seconds" % (time.time() - start_time))
#Total execution time: 28.463318347930908 seconds