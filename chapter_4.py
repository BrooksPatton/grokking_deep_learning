def neural_network(input, weight):
    return input * weight


input = 0.5
weight = 0.2
expected_prediction = 1.5
alpha = 0.01

while True:
    prediction = neural_network(input, weight)
    error = (prediction - expected_prediction) ** 2
    error_delta = (prediction - expected_prediction) * input
    weight = weight - (error_delta * alpha)
    print(error, prediction)
