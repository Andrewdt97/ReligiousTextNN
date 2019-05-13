from keras import utils
from processing.data_prep import initializeData
import numpy

documents, tokenizer = initializeData('..\data')

labels = utils.to_categorical(labels, len(set(labels)))
unique, counts = numpy.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))