'''
Andrew Thomas
CS 344 Final
'''
from keras import utils
from processing.data_prep import initializeData
import numpy

#########
# Entirely testing material.
##########
documents, tokenizer = initializeData('..\data')

labels = utils.to_categorical(labels, len(set(labels)))
unique, counts = numpy.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))