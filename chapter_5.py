def vector_multiplication(vector_1, vector_2):
    assert(len(vector_1) == len(vector_2))

    result = []

    for index in range(len(vector_1)):
        result.append(vector_1[index] * vector_2[index])

    return result


def weighted_sum(input, weights):
    return sum(vector_multiplication(input, weights))


def scalar_multiplication(scalar, vector):
    result = []

    for item in vector:
        result.append(scalar * item)

    return result


def neural_network(input, weights):
    return weighted_sum(input, weights)


input = [8.5, 0.65, 1.2]
weights = [0.1, 0.2, -0.1]
correct_prediction = 1023
alpha = 0.01

for _ in range(9):
    prediction = neural_network(input, weights)
    prediction_delta = prediction - correct_prediction
    error = prediction_delta ** 2
    weighted_prediction_deltas = scalar_multiplication(prediction_delta, input)

    for index in range(len(weights)):
        weights[0] = weights[0] - (weighted_prediction_deltas[index] * alpha)
