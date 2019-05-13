from processing.scrape_folder import folderScrape
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import numpy

import os

def initializeData(dataPath):
    listOfDocs = getData(dataPath)
    tk = Tokenizer()
    for doc in listOfDocs:
        tk.fit_on_texts(doc.text)
    return listOfDocs, tk


def getData(dataPath):
    listOfDocs = []
    for root, dirs, files in os.walk(dataPath):
        for dir in dirs:
            listOfDocs = listOfDocs + folderScrape(os.path.join(root, dir))
    return listOfDocs



def encodeData(texts, labels, tokenizer):
    encoded_docs = tokenizer.texts_to_matrix(texts, 'binary')
    unique, counts = numpy.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))
    encoded_labels = encodeLabels(labels)

    return (encoded_docs, encoded_labels)

def encodeLabels(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    encodedLabels = encoder.transform(labels)
    return encodedLabels

def extractTextAndLabels(listOfDocs):
    texts = []
    labels = []
    for doc in listOfDocs:
        texts.append(doc.text)
        labels.append(doc.label)
    return texts, labels
