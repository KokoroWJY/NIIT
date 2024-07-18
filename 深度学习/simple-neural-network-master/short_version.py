from numpy import exp, array, random, dot

# 定义一个四行五列的数组
training_set_inputs = array([[0, 0, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 0, 1, 1, 1],
                             [0, 1, 1, 1, 1]])

# 定义一个4行1列
training_set_outputs = array([[0, 1, 1, 0]]).T

random.seed(1)
synaptic_weights = 2 * random.random((5, 1)) - 1  # 初始权重
for iteration in range(30000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print(1 / (1 + exp(-(dot(array([1, 0, 0, 1, 1]), synaptic_weights)))))
