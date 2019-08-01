import numpy

weights = numpy.array([0.5, 0.48, 0-.7])
alpha = 0.1
streetlights = numpy.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1]
])
people_walked = numpy.array([0, 1, 0, 1, 1, 0])

current_input = streetlights[0]
goal_prediction = people_walked[0]

for _ in range(20):
    prediction = current_input.dot(weights)
    error = (prediction - goal_prediction) ** 2
    delta = prediction - goal_prediction
    weighted_delta = delta * current_input
    weights = weights - (alpha * weighted_delta)

    print('error', error)
    print('prediction', prediction)
    print('goal prediction', goal_prediction)
