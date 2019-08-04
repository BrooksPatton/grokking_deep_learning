from random import random

streetlight_patterns_training_inputs = []
streetlight_patterns_training_outputs = []
streetlight_patterns_testing_inputs = []
streetlight_patterns_testing_outputs = []

for _ in range(1000):
    training_left = round(random())
    training_center = round(random())
    training_right = round(random())
    testing_left = round(random())
    testing_center = round(random())
    testing_right = round(random())

    streetlight_patterns_training_inputs.append(
        [training_left, training_center, training_right])
    streetlight_patterns_training_outputs.append(
        1 if training_center == 1 else 0)
    streetlight_patterns_testing_inputs.append(
        [testing_left, testing_center, testing_right])
    streetlight_patterns_testing_outputs.append(
        1 if testing_center == 1 else 0)
