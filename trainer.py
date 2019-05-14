from keras import models, layers, utils
from sklearn.model_selection import train_test_split
import sys

def getModel(documents, labels):
    labels = utils.to_categorical(labels, len(set(labels)))
    trainText, testText , trainLabels, testLabels = train_test_split(documents, labels, test_size = 0.20)

    model = models.Sequential()
    model.add(layers.Embedding(5000, 128, input_length=trainText.shape[1]))
    model.add(layers.LSTM(64, return_sequences=True, dropout=0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.fit(trainText, trainLabels, epochs=15, batch_size=15)

    loss, accuracy = model.evaluate(testText, testLabels, verbose=1)
    print('Training Accuracy is {}'.format(accuracy*100))

    return model

# if __name__ == '__main__':
#     getModel(sys.argv[1])