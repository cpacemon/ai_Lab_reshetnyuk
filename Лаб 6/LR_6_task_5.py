import numpy as np
import neurolab as nl

# Б О Р
target = [[1, 1, 1, 1, 1,
           1, 0, 0, 1, 0,
           1, 1, 1, 1, 0,
           1, 0, 0, 1, 0,
           1, 1, 1, 1, 1],

          [0, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           0, 1, 1, 1, 0],

          [1, 1, 1, 1, 0,
           1, 0, 0, 1, 0,
           1, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1]]

chars = ['Р', 'Д', 'Р']
target = np.asfarray(target)
target[target == 0] = -1

# Create and train network
net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(output)):
    print(chars[i], (output[i] == target[i]).all())

print("Test of defaced Б:")
test = np.asfarray([1, 0, 1, 1, 1,
                    1, 0, 0, 1, 0,
                    1, 1, 1, 1, 0,
                    1, 0, 0, 1, 0,
                    1, 1, 0, 1, 1])
test[test == 0] = -1
output = net.sim([test])
print((output[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))

print("Test of defaced О:")
test = np.asfarray([1, 1, 1, 1, 0,
                    1, 0, 0, 0, 1,
                    1, 0, 0, 0, 1,
                    1, 0, 0, 0, 1,
                    0, 1, 1, 1, 0])
test[test == 0] = -1
output = net.sim([test])
print((output[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))

print("Test of defaced Р:")
test = np.asfarray([1, 1, 1, 1, 0,
                    1, 0, 0, 1, 0,
                    1, 1, 1, 1, 0,
                    1, 0, 0, 1, 1,
                    1, 0, 1, 0, 1])
test[test == 0] = -1
output = net.sim([test])
print((output[0] == target[2]).all(), 'Sim. steps', len(net.layers[0].outs))
