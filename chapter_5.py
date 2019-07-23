def neural_network(input, weights):
    return scalar_vector_multipy(input, weights)


def scalar_vector_multipy(scalar, vector):
    result = []

    for number in vector:
        result.append(scalar * number)

    return result


input = 0.65
goals = [0.1, 1, 0.1]
alpha = 0.1
weights = [0.3, 0.2, 0.9]

for _ in range(200):

    # get the predictions
    predictions = neural_network(input, weights)

    # get the prediction deltas
    prediction_deltas = map(
        lambda prediction, goal: prediction - goal, predictions, goals)

    # get the weighted errors
    weighted_error_deltas = scalar_vector_multipy(input, prediction_deltas)

    # change all the weights
    weights = list(map(lambda weight, derivative: weight -
                       (derivative * alpha), weights, weighted_error_deltas))

    print(predictions, weights)
