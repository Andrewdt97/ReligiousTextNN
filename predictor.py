'''
Andrew Thomas
CS 344 Final
'''
from trainer import getModel
from processing.data_prep import getDocuments, getTokenizer, extractTextAndLabels, encodeData
import numpy as np
from keras import utils


print('Getting intra texts')
intraDocs = getDocuments('./data/intra')

print('Creating tokenizer')
tokenizer = getTokenizer([intraDocs])

print('Encoding data')
intraTexts, intraLabels = extractTextAndLabels(intraDocs)
encodedIntraTexts, encodedIntraLabels = encodeData(intraTexts, intraLabels, tokenizer)

print('Training holy texts')
intraModel = getModel(encodedIntraTexts, encodedIntraLabels)
