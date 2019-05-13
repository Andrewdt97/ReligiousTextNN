from keras import models, layers, utils
from sklearn.model_selection import train_test_split
import sys

def getModel(documents, labels, tokenizer):
    labels = utils.to_categorical(labels, len(set(labels)))
    trainText, testText , trainLabels, testLabels = train_test_split(documents, labels , test_size = 0.20)

    model = models.Sequential()
    model.add(layers.Embedding(len(tokenizer.word_counts), 64, input_length=len(tokenizer.word_counts)+1))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.fit(trainText, trainLabels, epochs=10, batch_size=15)

    loss, accuracy = model.evaluate(testText, testLabels, verbose=1)
    print('Training Accuracy is {}'.format(accuracy*100))

    return model

# if __name__ == '__main__':
#     getModel(sys.argv[1])