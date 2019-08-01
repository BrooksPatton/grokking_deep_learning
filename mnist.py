import deep_learning_functions as dl

inputs = [
    [8.5, 0.65, 1.2],
    [9.5, 0.8, 1.3],
    [9.9, 0.8, 0.5],
    [9.0, 0.9, 1.0]
]
outputs = [
    [0.1, 1, 0.1],
    [0.0, 1, 0],
    [0, 0, 0.1],
    [0.1, 1, 0.2]
]
testing_inputs = [
    [9, 0.9, 1.2],
    [10, 0.88, 1.3],
    [9.9, 0.5, 0.5],
    [9.5, 0.75, 1.5]
]
testing_outputs = [
    [0.1, 1, 0.1],
    [0.0, 1, 0],
    [0, 0, 0.1],
    [0.1, 1, 0.2]
]
alpha = 0.01

dl.train(inputs, outputs, )
