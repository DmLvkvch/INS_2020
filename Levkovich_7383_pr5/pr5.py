import csv

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import matplotlib.pyplot as plt

SIZE_TRAIN_DATA = 10000
SIZE_TEST_DATA = 500

def genData(size):
    data = []
    targets = []
    for i in range(size):
        X = np.random.normal(-5, 10)
        e = np.random.normal(0, 0.3)
        data.append((-np.power(X, 3) + 3, np.log(np.abs(X)) + e, np.exp(X) + e, X + 4 + e,-X + np.sqrt(np.abs(X)) + e , X + e))
        targets.append((np.sin(3*X) + e))
    return data, targets


def write_csv(path, data):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',',
                        quoting=csv.QUOTE_MINIMAL)
        try:
            for item in data:
                writer.writerow(item)
        except Exception as ex:
            writer.writerows(map(lambda x: [x], data))


train_data, train_targets = genData(SIZE_TRAIN_DATA)
test_data, test_targets = genData(SIZE_TEST_DATA)
write_csv('train_data.csv', np.round(train_data, 4))
write_csv('train_targets.csv', np.round(train_targets, 4))
write_csv('test_data.csv', np.round(test_data, 4))
write_csv('test_targets.csv', np.round(test_targets, 4))

mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std

# encode
main_input = Input(shape=(6,))

encoded = Dense(64, activation='relu')(main_input)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(6, activation='relu')(encoded)

# decode
input_dec = Input(shape=(6,))

decoded = Dense(32, activation='relu', name='layer1')(encoded)
decoded = Dense(64, activation='relu', name='layer2')(decoded)
decoded = Dense(60, activation='relu', name='layer3')(decoded)
decoded = Dense(6, name="layer4")(decoded)
# regression



#for i in range(64):
predicted = Dense(100, activation='relu')(encoded)
predicted = Dense(100, activation='relu')(predicted)
predicted = Dense(100, activation='relu')(predicted)
#predicted = Dense(16, activation='relu')(predicted)
predicted = Dense(1)(predicted)
encoder = Model(main_input, encoded)
regr_model = Model(main_input, predicted)
autoencoder = Model(main_input, decoded)
autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])
autoencoder.fit(train_data, train_data, epochs=500, batch_size=128, shuffle=True,
                    validation_data=(test_data, test_data))
encode_data = encoder.predict(test_data)

decoder = autoencoder.get_layer('layer1')(input_dec)
decoder = autoencoder.get_layer('layer2')(decoder)
decoder = autoencoder.get_layer('layer3')(decoder)
decoder = autoencoder.get_layer('layer4')(decoder)
decoder = Model(input_dec, decoder)
decode_data = decoder.predict(encode_data)

regr_model.compile(optimizer="adam", loss="mse", metrics=['mae'])
History = regr_model.fit(train_data, train_targets, epochs=500, batch_size=128,
                           validation_data=(test_data, test_targets))
predict_data = regr_model.predict(test_data)
loss = History.history['loss']
v_loss = History.history['val_loss']
x = range(1, 501)
plt.plot(x, loss, 'c', label='Train')
plt.plot(x, v_loss, 'g', label='Test')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.show()
plt.clf()
decoder.save('decoder.h5')
encoder.save('encoder.h5')
regr_model.save('regression.h5')
write_csv('encoded_data.csv', np.round(encode_data, 4))
write_csv('decoded_data.csv', np.round(decode_data, 4))
write_csv('predicted_data.csv', np.round(predict_data, 4))