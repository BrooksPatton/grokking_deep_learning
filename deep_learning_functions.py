def initialize_weights(inputs, outputs):
    matrix_width = len(inputs)
    matrix_height = len(outputs)
    weights = []

    for _ in range(matrix_height):
        weights.append([0.5] * matrix_width)

    return weights


def neural_network(inputs, outputs, weights, alpha, testing_inputs, testing_outputs):
    for index in range(len(inputs)):
        current_inputs = inputs[index]
        current_outputs = outputs[index]
        predictions = []

        for weight in weights:
            predictions.append(dot(current_inputs, weight))

        # errors = calculate_errors(predictions, current_outputs)
        deltas = calculate_deltas(predictions, current_outputs)
        weighted_deltas = outer_product(deltas, current_inputs)
        limited_weighted_deltas = scalar_matrix_multiply(
            alpha, weighted_deltas)
        weights = matrix_subtract(weights, limited_weighted_deltas)

    return weights


def outer_product(vector_1, vector_2):
    matrix = []

    for vector_1_value in vector_1:
        row = []

        for vector_2_value in vector_2:
            row.append(vector_1_value * vector_2_value)

        matrix.append(row)

    return matrix


def scalar_matrix_multiply(scalar, matrix):
    result = []

    for row in matrix:
        new_row = []

        for value in row:
            new_row.append(scalar * value)

        result.append(new_row)

    return result


def dot(vector_1, vector_2):
    result = 0

    for index in range(len(vector_1)):
        result = result + (vector_1[index] * vector_2[index])

    return result


def calculate_errors(predictions, outputs):
    errors = calculate_deltas(predictions, outputs)

    for index in range(len(errors)):
        errors[index] = errors[index] ** 2

    return errors


def calculate_deltas(predictions, outputs):
    deltas = []

    for index in range(len(predictions)):
        deltas.append(predictions[index] - outputs[index])

    return deltas


def check_neural_network(inputs, outputs, weights):
    times_correct = 0

    for index in range(len(inputs)):
        current_inputs = inputs[index]
        current_outputs = outputs[index]
        predictions = []

        for weight_row in weights:
            predictions.append(dot(current_inputs, weight_row))

        prediction_index = max_value_in_array(predictions)

        if(current_outputs[prediction_index] == 1):
            times_correct = times_correct + 1

    return times_correct / len(inputs)


def matrix_subtract(matrix_1, matrix_2):
    result = []

    for height_index in range(len(matrix_1)):
        new_row = []

        for width_index in range(len(matrix_1[height_index])):
            new_row.append(
                matrix_1[height_index][width_index] - matrix_2[height_index][width_index])

        result.append(new_row)

    return result


def max_value_in_array(array):
    max_value_index = 0
    max_value = 0

    for index in range(len(array)):
        if(array[index] > max_value):
            max_value = array[index]
            max_value_index = index

    return max_value_index


def train(training_inputs, training_outputs, testing_inputs, testing_outputs, alpha):
    for count in range(1000):
        if(count % 100 == 0):
            weights = initialize_weights(
                training_inputs[0], training_outputs[0])
            weights = neural_network(
                training_inputs, training_outputs, weights, alpha, testing_inputs, testing_outputs)
            accuracy = check_neural_network(
                testing_inputs, testing_outputs, weights)

            print(accuracy)
        if(accuracy >= 0.7):
            print('accuracy', accuracy)
            break

# Testing


def test_initialize_weights():
    inputs = [1] * 5
    outputs = [3] * 3
    expected_result = [
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5]
    ]

    assert(initialize_weights(inputs, outputs) == expected_result)


def test_outer_product():
    inputs = [1, 2, 3, 4, 5]
    deltas = [6, 7, 8]
    expected_result = [
        [6, 12, 18, 24, 30],
        [7, 14, 21, 28, 35],
        [8, 16, 24, 32, 40]
    ]

    assert(outer_product(deltas, inputs) == expected_result)


def test_scalar_matrix_multiply():
    scalar = 2
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    expected_result = [
        [2, 4, 6],
        [8, 10, 12],
        [14, 16, 18]
    ]

    assert(scalar_matrix_multiply(scalar, matrix) == expected_result)


def test_dot():
    vector_1 = [1, 2, 3]
    vector_2 = [4, 5, 6]
    expected_result = 32

    assert(dot(vector_1, vector_2) == expected_result)


def test_calculate_errors():
    predictions = [1, 2, 3]
    outcomes = [10, 11, 12]
    expected_result = [81, 81, 81]

    assert(calculate_errors(predictions, outcomes) == expected_result)


def test_calculate_deltas():
    predictions = [10, 20, 30]
    outcomes = [5, 6, 7]
    expected_result = [5, 14, 23]

    assert(calculate_deltas(predictions, outcomes) == expected_result)


def test_matrix_subtract():
    matrix_1 = [
        [100, 99, 98],
        [97, 96, 95],
        [94, 93, 92]
    ]
    matrix_2 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    expected_result = [
        [99, 97, 95],
        [93, 91, 89],
        [87, 85, 83]
    ]

    assert(matrix_subtract(matrix_1, matrix_2) == expected_result)


def test_max_value_in_array():
    array = [1, 9, 2, 7, 4]
    expected_result = 1

    assert(max_value_in_array(array) == expected_result)
