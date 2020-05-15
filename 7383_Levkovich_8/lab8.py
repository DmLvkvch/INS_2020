import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import keras.callbacks
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

def generate(model):
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    f = open('c.txt', 'a')
    f.write("Seed:")
    print("Seed:")
    f.write("\"" + ''.join([int_to_char[value] for value in pattern])+ "\"\n")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    f.close()
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        f = open('c.txt', 'a')
        f.write(result)
        f.close()
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

class CustomCB(keras.callbacks.Callback):
    def __init__(self, epochs):
        super(CustomCB, self).__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs={}):
        if epoch in self.epochs:
            f = open('c.txt', 'a')
            f.write("Epoch: "+str(epoch))
            f.close()
            print(epoch)
            generate(model)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, CustomCB([1, 2, 5, 7, 11, 15, 19])]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
generate(model)
model.save('alice.h5')