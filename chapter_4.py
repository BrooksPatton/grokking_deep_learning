weight = 0.1
learning_speed = 0.01


def neural_network(input, weight):
    prediction = input * weight
    return prediction


number_of_toes = [8.5]
won_game = [1]
input = number_of_toes[0]
goal_prediction = won_game[0]

for _ in range(10):
    prediction = neural_network(input, weight)
    error = (prediction - goal_prediction) ** 2
    error_delta = prediction - goal_prediction
    possible_new_weight = error_delta * input
    weight = weight - (possible_new_weight * learning_speed)
