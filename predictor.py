from trainer import getModel
from processing.data_prep import getDocuments, getTokenizer, extractTextAndLabels, encodeData
import numpy as np
from keras import utils

# def getModel(dataPath):
#
#     documents, tokenizer = initializeData(dataPath)
#
# with open('inputText.txt', 'r') as file:
#     stringToPredict = file.read()
# stringToPredict = cleanse(stringToPredict)
# tokenizer.fit_on_texts(stringToPredict)
#
# plainTexts, plainLabels = extractTextAndLabels(documents)
# # print('dohicky')
# # print(plainLabels)
# encodedTexts, encodedLabels = encodeData(plainTexts, plainLabels, tokenizer)
#
# model = getModel(encodedTexts, encodedLabels, tokenizer)
#
# newLabel = model.predict_classes([tokenizer.texts_to_matrix(stringToPredict)])
# print(newLabel)

# def compare(predictions, labels):
#     total = correct = 0
#     for i in range(len(predictions):
#         total += 1
#         if (predictions[i] == labels[i]):
#             correct += 1
#     return correct, total

print('Getting holy texts')
holyTexts = getDocuments('./data/holy_texts')

print('Getting writings')
writings = getDocuments('./data/writings')

print('Creating tokenizer')
corpuses = [holyTexts, writings]
tokenizer = getTokenizer(corpuses)

print('Encoding data')
htTexts, htLabels = extractTextAndLabels(holyTexts)
encodedHtTexts, encodedHtLabels = encodeData(htTexts, htLabels, tokenizer)

wTexts, wLabels = extractTextAndLabels(writings)
encodedWTexts, encodedWLabels = encodeData(wTexts, wLabels, tokenizer)

print('Training holy texts')
holyTextsModel = getModel(encodedHtTexts, encodedHtLabels)

print('Training writings model')
writingsModel = getModel(encodedWTexts, encodedWLabels)

print('\n\n\n')
loss, accuracy = holyTextsModel.evaluate(encodedWTexts, utils.to_categorical(encodedWLabels, len(set(encodedWLabels))), verbose=1)
print('Cross Accuracy is {}'.format(accuracy*100))
# predictWritingsOnHt = holyTextsModel.predict(np.array(encodedWTexts))
# print(predictWritingsOnHt)
# predictedClasses = np.argmax(predictWritingsOnHt, axis=1)
# predictedClasses = predictedClasses.tolist()
# correct, total = compare(predictedClasses, encodedWLabels)
# print(predictedClasses)