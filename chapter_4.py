weight = 0.1
learn_rate = 0.01


def neural_network(input, weight):
    prediction = input * weight
    return prediction


number_of_toes = [8.5]
win_or_lose_binary = [1]
input = number_of_toes[0]
expected_prediction = win_or_lose_binary[0]
prediction = neural_network(input, weight)
error = (prediction - expected_prediction) ** 2

print(error)

prediction_up = neural_network(input, weight + learn_rate)
error_up = (prediction_up - expected_prediction) ** 2

prediction_down = neural_network(input, weight - learn_rate)
error_down = (prediction_down - expected_prediction) ** 2

if(error > error_down or error > error_up):
    if(error_down < error_up):
        weight = weight - learn_rate
    if(error_up < error_down):
        weight = weight + learn_rate
