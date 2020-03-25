import numpy as np
from keras.layers import Dense
from keras.models import Sequential


#
def element_by_element(data, weights):
    result = data.copy()  # prevent from changes
    for x in range(len(weights)):
        tmp = np.zeros((len(result), len(weights[x][1])))
        for i in range(len(result)):
            for j in range(len(weights[x][1])):
                s = 0
                for k in range(len(result[i])):
                    s += result[i][k] * weights[x][0][k][j]
                if x == len(weights) - 1:
                    tmp[i][j] = sigmoid(s + weights[x][1][j])
                else:
                    tmp[i][j] = relu(s + weights[x][1][j])
        result = tmp
    return result


def tensor_numpy(data, weights):
    result = data.copy()  # prevent from changes
    for i in range(len(weights)):
        if i == len(weights)-1:
            result = sigmoid((np.dot(result, weights[i][0]) + weights[i][1]))
        else:
            result = relu(np.dot(result, weights[i][0]) + weights[i][1])
    return result


def relu(x):
    return np.maximum(x, 0.)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logic_expression(expression):  # != xor
    return int((expression[0] or expression[1]) ^ (not (expression[1] and expression[2])))


def evaluate(matrix):
    result = []
    for i in matrix:
        result.append(logic_expression(i))
    return result


def test(model, data):
    weights = []
    for layer in model.layers:  # model.layers is a flattened list of the layers comprising the model
        weights.append(layer.get_weights())  # layer.get_weights(): returns the weights of the layer
        # Numpy arrays.
    correct_answer = evaluate(data)
    element_by_element_answer = element_by_element(data, weights)
    tensor_numpy_answer = tensor_numpy(data, weights)
    model_answer = model.predict(data)
    return correct_answer, element_by_element_answer, tensor_numpy_answer, model_answer


def gen_train_data():
    binary = []
    for i in range(0, 8):
        i = "{0:03b}".format(i)
        tmp = [int(i[0]), int(i[1]), int(i[2])]
        binary.append(tmp)
    data = np.array(binary)
    return data


train_data = gen_train_data()
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(3,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
(correct_answer, element_by_element_answer, tensor_numpy_answer, model_answer) = test(model, train_data)
print(correct_answer)
print(element_by_element_answer)
print(tensor_numpy_answer)
print(model_answer)
model.fit(train_data, evaluate(train_data), epochs=150, batch_size=1)
(correct_answer, element_by_element_answer, tensor_numpy_answer, model_answer) = test(model, train_data)
print(correct_answer)
print(element_by_element_answer)
print(tensor_numpy_answer)
print(model_answer)
