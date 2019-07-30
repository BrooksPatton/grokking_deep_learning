def test_concatentate_lists():
    lists = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    expected_output = [1, 2, 3, 4, 5, 6]
    assert(concatenate_lists(lists) == expected_output)


def concatenate_lists(lists):
    result = []
    for single_list in lists:
        result.extend(single_list)
    return result


def test_convert_mnist_images_to_lists():
    mnist_images = [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [7, 8, 9],
            [10, 11, 12]
        ]
    ]

    expected_output = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12]
    ]

    assert(convert_mnist_images_to_lists(mnist_images) == expected_output)


def convert_mnist_images_to_lists(mnist_images):
    converted_images = []

    for mnist_image in mnist_images:
        converted_images.append(concatenate_lists(mnist_image))

    return converted_images


def test_convert_number_to_decimal_output():
    label = 1
    expected_output = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 2
    expected_output = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 3
    expected_output = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 4
    expected_output = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 5
    expected_output = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 6
    expected_output = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 7
    expected_output = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 8
    expected_output = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 9
    expected_output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert(convert_number_to_decimal_output(label) == expected_output)

    label = 0
    expected_output = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert(convert_number_to_decimal_output(label) == expected_output)


def convert_number_to_decimal_output(number):
    output = [0] * 10

    output[number] = 1

    return output


def test_create_zeros_matrix():
    x = 5
    y = 3
    expected_output = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    assert(create_zeros_matrix(x, y) == expected_output)


def create_zeros_matrix(width, height):
    matrix = []

    for _ in range(height):
        matrix.append([0] * width)

    return matrix


def test_dot_vector():
    vector_1 = [1, 2, 3]
    vector_2 = [4, 5, 6]
    expected_output = 32
    assert(dot_vector(vector_1, vector_2) == expected_output)


def dot_vector(vector_1, vector_2):
    assert(len(vector_1) == len(vector_2))
    result = 0
    for index in range(len(vector_1)):
        result = result + (vector_1[index] * vector_2[index])
    return result


def test_calculate_error():
    predictions = [1, 2, 3, 4, 5]
    correct_outcomes = [6, 7, 8, 9, 10]
    expected_output = [25, 25, 25, 25, 25]
    assert(calculate_error(predictions, correct_outcomes) == expected_output)


def test_calculate_error_using_more_realistic_data():
    predictions = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    correct_outcomes = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_output = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert(calculate_error(predictions, correct_outcomes) == expected_output)


def calculate_error(predictions, expected_outcomes):
    assert(len(predictions) == len(expected_outcomes))
    errors = []
    for index in range(len(predictions)):
        errors.append((predictions[index] - expected_outcomes[index]) ** 2)
    return errors


def test_calculate_delta():
    predictions = [100, 90, 80]
    expected_outcomes = [10, 5, 1]
    expected_result = [90, 85, 79]
    assert(calculate_delta(predictions, expected_outcomes) == expected_result)


def calculate_delta(predictions, expected_predictions):
    assert(len(predictions) == len(expected_predictions))

    results = []

    for index in range(len(predictions)):
        results.append(predictions[index] - expected_predictions[index])

    return results


def test_outer_product():
    inputs = [1, 2, 3, 4, 5]
    deltas = [6, 7, 8]
    expected_outcomes = [
        [6, 12, 18, 24, 30],
        [7, 14, 21, 28, 35],
        [8, 16, 24, 32, 40]
    ]
    assert(outer_product(inputs, deltas) == expected_outcomes)


def outer_product(vector_1, vector_2):
    matrix = create_zeros_matrix(len(vector_1), len(vector_2))

    for vector_2_index in range(len(vector_2)):
        for vector_1_index in range(len(vector_1)):
            matrix[vector_2_index][vector_1_index] = vector_2[vector_2_index] * \
                vector_1[vector_1_index]

    return matrix

def test_scalar_matrix_multiply():
    scalar = 2
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
    expected_outcomes = [
        [2, 4, 6, 8],
        [10, 12, 14, 16]
    ]
    assert(scalar_matrix_multiply(scalar, matrix) == expected_outcomes)

def scalar_matrix_multiply(scalar, matrix):
    result = create_zeros_matrix(len(matrix[0]), len(matrix))

    for y_index in range(len(matrix)):
        for x_index in range(len(matrix[0])):
            result[y_index][x_index] = matrix[y_index][x_index] * scalar

    return result

def test_matrix_subtract():
    matrix_1 = [
        [100, 90, 80, 70],
        [60, 50, 40, 30]
    ]
    matrix_2 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
    expected_outcomes = [
        [99, 88, 77, 66],
        [55, 44, 33, 22]
    ]

    assert(matrix_subtract(matrix_1, matrix_2) == expected_outcomes)

def matrix_subtract(matrix_1, matrix_2):
    result = create_zeros_matrix(len(matrix_1[0]), len(matrix_1))

    for y_index in range(len(matrix_1)):
        for x_index in range(len(matrix_1[0])):
            result[y_index][x_index] = matrix_1[y_index][x_index] - matrix_2[y_index][x_index]
    
    return result

def train(all_inputs, weights, alpha, all_expected_outcomes):
    for index in range(len(all_inputs)):
        inputs = all_inputs[index]
        expected_outcomes = all_expected_outcomes[index]
        prediction = neural_network(inputs, weights)
        errors = calculate_error(prediction, expected_outcomes)
        deltas = calculate_delta(prediction, expected_outcomes)
        weighted_deltas = outer_product(inputs, deltas)
        scaled_weighted_deltas = scalar_matrix_multiply(alpha, weighted_deltas)
        weights = matrix_subtract(weights, scaled_weighted_deltas)

def neural_network(inputs, weights):
    prediction = []
    for weight_row in weights:
        prediction.append(dot_vector(inputs, weight_row))
    return prediction


if __name__ == "__main__":
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    raw_mnist_images = x_train[0:1000]
    mnist_images = convert_mnist_images_to_lists(raw_mnist_images)
    mnist_labels = y_train[0:1000]
    correct_outputs = []

    for mnist_label in list(mnist_labels):
        correct_outputs.append(
            convert_number_to_decimal_output(mnist_label))

    weights = create_zeros_matrix(
        len(mnist_images[0]), len(correct_outputs[0]))
    alpha = 0.0000001

    train(mnist_images, weights, alpha, correct_outputs)
