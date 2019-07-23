def neural_network(input_data, weights):
    assert(len(input_data) == len(weights))

    prediction = []

    for weight in weights:
        prediction.append(weighted_sum(input_data, weight))
    
    return prediction

def weighted_sum(vector_1, vector_2):
    assert(len(vector_1) == len(vector_2))

    result = 0

    for index in range(len(vector_1)):
        result = result + (vector_1[index] * vector_2[index])

    return result

def calculate_deltas(prediction, true):
    assert(len(prediction) == len(true))

    deltas = []

    for index in range(len(prediction)):
        deltas.append(prediction[index] - true[index])

    return deltas

def calculate_errors(prediction, true):
    assert(len(prediction) == len(true))

    errors = []

    for index in range(len(prediction)):
        errors.append((prediction[index] - true[index]) ** 2)

    return errors

# def calculate_weighted_deltas(deltas, input_data):
#     assert(len(deltas) == len(input_data))

#     weighted_deltas = []

#     for item in input_data:
#         weighted_deltas.append(scalar_vector_multiply(item, deltas))

#     return weighted_deltas

def testing():
    print('testing weighted_sum')
    assert(weighted_sum([1, 2, 3], [4, 5, 6]) == 32)

    print('testing neural_network')
    input_data = [1, 2, 3]
    weights = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    expected_outcome = [14, 32, 50]
    assert(neural_network(input_data, weights) == expected_outcome)

    print('testing calculate_deltas')
    prediction = [1, 2, 3]
    true = [4, 5, 6]
    expected_outcome = [-3, -3, -3]
    assert(calculate_deltas(prediction, true) == expected_outcome)

    print('testing calculate_errors')
    prediction = [1, 2, 3]
    true = [4, 5, 6]
    expected_outcome = [9, 9, 9]
    assert(calculate_errors(prediction, true) == expected_outcome)

    print('testing calculate_weighted_deltas')
    deltas = [1, 2, 3]
    input_data = [4, 5, 6]
    expected_outcome = [
        [4, 8, 12],
        [5, 10, 15],
        [6, 12, 18]
    ]
    # assert(calculate_weighted_deltas(deltas, input_data) == expected_outcome)

testing()

weights = [
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]
]

toes = [8.5, 9.5, 9.9, 9.0]
win_loss_record = [0.65, 0.8, 0.8, 0.9]
number_of_fans = [1.2, 1.3, 0.5, 1.0]

hurt = [0.1, 0.0, 0.0, 0.1]
win = [1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]

alpha = 0.01

input_data = [toes[0], win_loss_record[0], number_of_fans[0]]
true = [hurt[0], win[0], sad[0]]

predictions = neural_network(input_data, weights)
deltas = calculate_deltas(predictions, true)
errors = calculate_errors(predictions, true)
# weighted_deltas = calculate_weighted_deltas(deltas, input)

# update_weights(weights, weighted_deltas, alpha)