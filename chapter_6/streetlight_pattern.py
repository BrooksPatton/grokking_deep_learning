import numpy as np

np.random.seed(1)


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

walk_vs_stop = np.array([[0, 1, 0, 1]]).T

test_streetlights = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
    [0, 0, 0]
])

test_streetlights_outputs = [0, 1, 0, 1, 1, 1, 0, 0]

alpha = 0.2
hidden_size = 4

weights_0_1 = 2*np.random.random((3, hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size, 1)) - 1

for iteration in range(600):
    layer_2_error = 0
    for i in range(len(streetlights)):
        test_correct = 0
        layer_0 = streetlights[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)

        layer_2_delta = (layer_2 - walk_vs_stop[i:i+1])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)

        weights_1_2 -= alpha * layer_1.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.dot(layer_1_delta)

        for test_index in range(len(test_streetlights_outputs)):
            layer_0 = test_streetlights[test_index:test_index+1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            if(round(layer_2[0][0] == test_streetlights_outputs[test_index])):
                test_correct = test_correct + 1

    if(iteration % 10 == 9):
        print("Error:" + str(layer_2_error))
        print('accuracy: ', test_correct / len(test_streetlights_outputs))
