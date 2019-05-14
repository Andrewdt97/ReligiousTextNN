'''
Andrew Thomas
CS 344 Final
'''
import os
import string
from classes.document import Document
import nltk


def cleanse(str):
    '''
    Cleanses a string of impurities
    '''
    # Filter out characters
    charsAllowed = set(string.ascii_letters + '-' + string.whitespace)
    filt = filter(lambda x: x in charsAllowed, str)
    tokenedString = ''.join(filt).split()

    # Remove stop words
    # try:
    #     nltk.data.find('corpora/stopwords')
    # except:
    #     nltk.download('stopwords')
    # stop_words = set(nltk.corpus.stopwords.words('english'))
    # filteredListOfWords = [w for w in tokenedString if not w in stop_words]

    return tokenedString

def prepFile(fileName):
    '''
    Seperates a file into it's texts and returns this as a list
    '''
    with open(fileName, 'r', encoding='UTF-8') as file:
        # Split based on delimeter $$$
        listOfDocs = file.read().split('$$$')

    for i in range(len(listOfDocs)):
        listOfDocs[i] = cleanse(listOfDocs[i])
    return listOfDocs

def folderScrape(folderPath):
    '''
    For a given folder, read all of the files within
    '''
    docList = []
    for root, dirs, files in os.walk(folderPath):
        for file in files:
            path = os.path.join(root, file)
            texts = prepFile(os.path.abspath(path))
            for text in texts:
                docList.append(Document(text, root))
    return docList
