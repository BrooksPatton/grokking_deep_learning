def weighted_sum(inputs, weights):
    assert(len(inputs) == len(weights))

    result = 0

    for index in range(len(inputs)):
        result = result + (inputs[index] * weights[index])

    return result


def neural_network(inputs, weights):
    return weighted_sum(inputs, weights)


def elemental_multiplication(number, vector):
    result = []

    for item in vector:
        result.append(item * number)

    return result


weights = [0.1, 0.2, -0.1]
alpha = 0.01

number_of_toes = [8.5, 9.5, 9.9, 9.0]
win_loss_percentage = [0.65, 0.8, 0.8, 0.9]
number_of_fans = [1.2, 1.3, 0.5, 1.0]

games_won_or_lost = [1, 1, 0, 1]
game_result = games_won_or_lost[0]
input = [number_of_toes[0], win_loss_percentage[0], number_of_fans[0]]

prediction = neural_network(input, weights)
error = (prediction - game_result) ** 2
error_delta = prediction - game_result
weighted_error_deltas = elemental_multiplication(error_delta, input)

for index in range(len(weights)):
    weights[index] = weights[index] - (alpha * weighted_error_deltas[index])
