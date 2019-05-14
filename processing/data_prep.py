'''
Andrew Thomas
CS 344 Final
'''
from processing.scrape_folder import folderScrape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy
import os


def getTokenizer(listOfCorps):
    '''
    Given a list of corpuses, generate the tokenizer with no more than 5000 words
    '''
    tk = Tokenizer(num_words=5000, lower=True)
    for corpus in listOfCorps:
        for doc in corpus:
            tk.fit_on_texts(doc.text)
    return tk

def getDocuments(dataPath):
    '''
    Given the path of the root folder, collect all data
    '''
    listOfDocs = []
    for root, dirs, files in os.walk(dataPath):
        for dir in dirs:
            listOfDocs = listOfDocs + folderScrape(os.path.join(root, dir))
    return listOfDocs


def encodeData(texts, labels, tokenizer):
    '''
    Given raw strings, labels, and a tokenizer, ecode the strings and labels
    '''
    # Encode documents as a sequence and make all sequences 4400 tokens long
    encoded_docs = tokenizer.texts_to_sequences(texts)
    encoded_docs = pad_sequences(encoded_docs, 4400, padding='post')

    # Format labels
    unique, counts = numpy.unique(labels, return_counts=True)
    print('\n!!!!!\nDocument count by label:')
    print(dict(zip(unique, counts)))
    encoded_labels = encodeLabels(labels)

    return (encoded_docs, encoded_labels)

def encodeLabels(labels):
    '''
    Translate string labels into float ones
    '''
    encoder = LabelEncoder()
    encoder.fit(labels)
    encodedLabels = encoder.transform(labels)
    return encodedLabels

def extractTextAndLabels(listOfDocs):
    '''
    Given a list of documents, return a tuple of their texts and labels
    '''
    texts = []
    labels = []
    for doc in listOfDocs:
        texts.append(doc.text)
        labels.append(doc.label)
    return texts, labels
