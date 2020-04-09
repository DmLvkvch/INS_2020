import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import pylab
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
import matplotlib
import numpy as np
from PIL import Image
from keras import optimizers

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()


train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def build_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model



def test_optimizer(optimizer, name):
    model = build_model()
   # optimiz = optimizers.RMSprop(rho=0.9)
    #optimizer = optimizers.RMSprop(rho=0.9999999)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=5,
                          batch_size=128, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    print('test_loss:', test_loss)#+" "+K.eval(model.optimizer.lr)
#    res_train_data.append(history.history['accuracy'][-1])
 #   res_test_data.append(test_acc)
    plt.title('Training and test accuracy '+name+" "+str(K.eval(model.optimizer.lr)))
    plt.plot(history.history['accuracy'], 'r', label='train')
    plt.plot(history.history['val_accuracy'], 'b', label='test')
    plt.legend()
    plt.savefig("rho" +str(K.eval(model.optimizer.lr))+"_acc"+str(K.eval(model.optimizer.lr)) + '.png')
    plt.clf()

    plt.title('Training and test loss '+name+" "+str(K.eval(model.optimizer.lr)))
    plt.plot(history.history['loss'], 'r', label='train')
    plt.plot(history.history['val_loss'], 'b', label='test')
    plt.legend()
    plt.savefig("rho" +str(K.eval(model.optimizer.lr))+"_loss"+str(K.eval(model.optimizer.lr)) + '.png')
    plt.clf()

def get_img(filename):
    image = Image.open(filename).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image/255
    return np.expand_dims(image, axis=0)


optimizerslist = [optimizers.Adam, optimizers.RMSprop,  optimizers.SGD, optimizers. Adagrad, optimizers.Adadelta]
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
res_train_data = []
res_test_data = []
for learn_rt in learning_rates:
    for optimizer in optimizerslist:
        test_optimizer(optimizer(learning_rate=learn_rt), optimizer.__name__)
    plt.bar(np.arange(len(optimizerslist)) * 3, res_train_data, width=1)
    plt.bar(np.arange(len(optimizerslist)) * 3 + 1, res_test_data, width=1)
    plt.xticks([3 * i + 0.5 for i in range(len(optimizerslist))])
    plt.title('learning_rate=' + str(learn_rt))
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.savefig('bar' + 'lr'+str(learn_rt) + '.png')
    plt.clf()
    res_test_data = []
    res_train_data = []