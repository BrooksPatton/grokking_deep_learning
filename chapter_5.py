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

def neural_network(input, weights):
    assert(len(input) == len(weights))
    return vector_matrix_multiply(input, weights)

weights = [
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]
]

alpha = 0.01
inputs = [8.5, 0.65, 1.2]
what_happened = [0.1, 1, 0.1]

predictions = neural_network(input, weights)
errors = list(map(lambda a, b: (a - b) ** 2, predictions, what_happened))
deltas = list(map(lambda a, b: a - b, predictions, what_happened))

# tests
test_weighted_sum()
test_vector_matrix_multiply()