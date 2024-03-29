'''
Andrew Thomas
CS 344 Final
'''
from keras import models, layers, utils
from sklearn.model_selection import train_test_split
import sys

def getModel(documents, labels):
	# Convert labels to one hot encoded arrays
    labels = utils.to_categorical(labels, len(set(labels)))
    # Turn 20% of the data into testing
    trainText, testText , trainLabels, testLabels = train_test_split(documents, labels, test_size = 0.20)

    model = models.Sequential()
    model.add(layers.Embedding(5000, 64, input_length=trainText.shape[1]))
    model.add(layers.LSTM(64, return_sequences=True, dropout=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.fit(trainText, trainLabels, epochs=5, batch_size=15)

    loss, accuracy = model.evaluate(testText, testLabels, verbose=1)
    print('Training Accuracy is {}'.format(accuracy*100))

    return model

# if __name__ == '__main__':
#     getModel(sys.argv[1])