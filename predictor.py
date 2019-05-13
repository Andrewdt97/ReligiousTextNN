from trainer import getModel
from processing.data_prep import initializeData, extractTextAndLabels, encodeData
from processing.scrape_folder import cleanse
import sys

documents, tokenizer = initializeData('.\data')

with open('inputText.txt', 'r') as file:
    stringToPredict = file.read()
stringToPredict = cleanse(stringToPredict)
tokenizer.fit_on_texts(stringToPredict)

plainTexts, plainLabels = extractTextAndLabels(documents)
# print('dohicky')
# print(plainLabels)
encodedTexts, encodedLabels = encodeData(plainTexts, plainLabels, tokenizer)

model = getModel(encodedTexts, encodedLabels, tokenizer)

newLabel = model.predict_classes([tokenizer.texts_to_matrix(stringToPredict)])
print(newLabel)