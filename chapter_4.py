weight = 0.5
input = 0.5
expected_prediction = 0.8
learn_rate = 0.001


def neural_network(input, weight):
    prediction = input * weight
    return prediction


for _ in range(1103):
    prediction = neural_network(input, weight)
    error = (prediction - expected_prediction) ** 2
    print("Error: " + str(error) + ' prediction: ' + str(prediction))

    up_prediction = neural_network(input, weight + learn_rate)
    up_error = (up_prediction - expected_prediction) ** 2
    down_prediction = neural_network(input, weight - learn_rate)
    down_error = (down_prediction - expected_prediction) ** 2

    if(error < up_error or error < down_error):
        if(up_error < down_error):
            weight = weight + learn_rate
        elif(down_error < up_error):
            weight = weight - learn_rate
