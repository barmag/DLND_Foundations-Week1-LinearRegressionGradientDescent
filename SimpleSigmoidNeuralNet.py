import numpy as np


def sigmoid(x):
    """Calculate the sigmoid of x"""
    sigmoidvalue = 1 / (1+np.exp(-1*x))
    return sigmoidvalue

def run():
    inputs = np.array([0.7, -0.3])
    weights = np.array([0.1, 0.7])
    bias = -0.1

    #output = 0
    #weightedInputs = (inputs * weights) + bias
    #output = sigmoid(np.sum(weightedInputs))
    output = sigmoid(np.dot(weights, inputs) + bias)
    print(output)

if __name__ == "__main__":
    run()
