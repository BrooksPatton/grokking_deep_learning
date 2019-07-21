knob_weight = 0.5
input = 0.5
goal_prediction = 0.8

prediction = input * knob_weight
error = (prediction - goal_prediction) ** 2

print(error)
