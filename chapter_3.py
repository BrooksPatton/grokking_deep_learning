def neural_network(input, weights):
    prediction = dot(input, weights)
    return prediction

def elementwise_multiplication(vector_1, vector_2):
    assert(len(vector_1) == len(vector_2))

    result = []

    for index in range(len(vector_1)):
        result.append(vector_1[index] * vector_2[index])
    
    return result

def elementwise_addition(vector_1, vector_2):
    assert(len(vector_1) == len(vector_2))

    result = []

    for index in range(len(vector_1)):
        result.append(vector_1[index] + vector_2[index])

    return result

def vector_sum(vector):
    return sum(vector)

def vector_average(vector):
    return sum(vector) / len(vector)

def dot(vector_1, vector_2):
    product = elementwise_multiplication(vector_1, vector_2)
    return vector_sum(product)

weights = [0.1, 0.2, 0]
number_of_toes = [8.5, 9.5, 10, 9]
win_loss_record = [0.65, 0.8, 0.8, 0.9]
number_of_fans = [1.2, 1.3, 0.5, 1.0]
input = [number_of_toes[0], win_loss_record[0], number_of_fans[0]]
prediction = neural_network(input, weights)
print(prediction)