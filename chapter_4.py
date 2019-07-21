weight = 0.5
input = 0.5
expected_prediction = 0.8


def neural_network(input, weight):
    prediction = input * weight
    return prediction


for _ in range(22):
    prediction = neural_network(input, weight)
    error = (prediction - expected_prediction) ** 2
    learn_rate = (prediction - expected_prediction) * input
    weight = weight - learn_rate
    print("Error: " + str(error) + ' prediction: ' + str(prediction))
