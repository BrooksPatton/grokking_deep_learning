def weighted_sum(vector_1, vector_2):
    assert(len(vector_1) == len(vector_2))
    result = 0
    for index in range(len(vector_1)):
        result = result + (vector_1[index] * vector_2[index])
    return result


def test_weighted_sum():
    print('testing weighted_sum')
    vector_1 = [1, 2, 3]
    vector_2 = [4, 5, 6]
    expected_result = 32
    assert(weighted_sum(vector_1, vector_2) == expected_result)


def test_vector_matrix_multiply():
    print('testing vector_matrix_multiply')
    vector = [1, 2, 3]
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    expected_result = [14, 32, 50]
    assert(vector_matrix_multiply(vector, matrix) == expected_result)


def vector_matrix_multiply(vector, matrix):
    assert(len(vector) == len(matrix))
    results = []
    for line in matrix:
        results.append(weighted_sum(vector, line))
    return results


def test_generate_weighted_deltas():
    print('testing generate_weighted_deltas')
    vector_1 = [1, 2, 3]
    vector_2 = [4, 5, 6]
    expected_result = [
        [4, 5, 6],
        [8, 10, 12],
        [12, 15, 18]
    ]
    assert(generate_weighted_deltas(vector_1, vector_2) == expected_result)


def generate_weighted_deltas(vector_1, vector_2):
    assert(len(vector_1) == len(vector_2))
    result = create_matrix(len(vector_1), len(vector_2))
    for index_y in range(len(vector_1)):
        for index_x in range(len(vector_2)):
            result[index_y][index_x] = vector_1[index_y] * vector_2[index_x]
    return result


def test_create_matrix():
    print('testing create_matrix')
    x = 3
    y = 3
    expected_result = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    assert(create_matrix(x, y) == expected_result)


def create_matrix(x, y):
    matrix = []
    for _ in range(y):
        matrix.append([0] * x)
    return matrix


def neural_network(inputs, weights):
    assert(len(inputs) == len(weights))
    return vector_matrix_multiply(inputs, weights)


def test_update_weights():
    print('testing update_weights')
    weighted_deltas = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    weights = [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ]
    alpha = -1
    expected_result = [
        [11, 22, 33],
        [44, 55, 66],
        [77, 88, 99]
    ]
    assert(update_weights(weighted_deltas, weights, alpha) == expected_result)


def update_weights(weighted_deltas, weights, alpha):
    assert(len(weighted_deltas) == len(weights))
    for index_y in range(len(weights)):
        assert(len(weighted_deltas[index_y]) == len(weights[index_y]))
        for index_x in range(len(weights[index_y])):
            weights[index_y][index_x] = weights[index_y][index_x] - \
                (weighted_deltas[index_y][index_x] * alpha)
    return weights


weights = [
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]
]

alpha = 0.01
inputs = [8.5, 0.65, 1.2]
what_happened = [0.1, 1, 0.1]

while True:
    predictions = neural_network(inputs, weights)
    errors = list(map(lambda a, b: (a - b) ** 2, predictions, what_happened))
    deltas = list(map(lambda a, b: a - b, predictions, what_happened))
    weighted_deltas = generate_weighted_deltas(deltas, inputs)
    weights = update_weights(weighted_deltas, weights, alpha)

    print(predictions)

    done_count = 0
    for error in errors:
        if(error < 0.00000000001):
            done_count = done_count + 1
    if(done_count == len(errors)):
        break

# tests
# test_weighted_sum()
# test_vector_matrix_multiply()
# test_create_matrix()
# test_generate_weighted_deltas()
# test_update_weights()
