import numpy as np
import neurolab as nl

target = [[-1, 1, -1, -1, 1, -1, -1, 1, -1],
          [1, 1, 1, 1, -1, 1, 1, -1, 1],
          [1, -1, 1, 1, 1, 1, 1, -1, 1],
          [1, 1, 1, 1, -1, -1, 1, -1, -1],
          [-1, -1, -1, -1, 1, -1, -1, -1, -1]]

input = [[-1, -1, 1, 1, 1, 1, 1, -1, 1],
         [-1, -1, 1, -1, 1, -1, -1, -1, -1],
         [-1, -1, -1, -1, 1, -1, -1, 1, -1]]

# Створення та тренування
net = nl.net.newhem(target)

output = net.sim(target)
print("Test on train data (must be [0, 1, 2, 3, 4]):")
print(np.argmax(output, axis=0))

output = net.sim([input[0]])
print("Outputs on recurrent cycle:")
print(np.array(net.layers[1].outs))

output = net.sim(input)
print("Test on test sample:")
print(output)