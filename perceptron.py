import random
import numpy as np

LEARNING_RATE = 1
# Epochs = Learning iterations; number of times the data shall be trained
EPOCHS = 10

class Perceptron():
    def __init__(self, X, expected):
        # Adding an extra index to the weight array
        # Index 0 is the bias, theta
        self.w = np.array([random.uniform(-0.5, 0.5) for i in range(len(X[0])+1)])

        print('Weights on start:')
        print(self.w , '\n')
        
        self.learn(X, expected)

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.w.T.dot(x)
        a = self.activation(z)
        return a
    
    def learn(self, X, expected):
        for i in range(EPOCHS):
            for j in range(len(expected)):
                x = np.insert(X[j], 0, 1)
                prediction = self.predict(x)
                error = expected[j] - prediction
                self.w = self.w + LEARNING_RATE * error * x
            print('Weights at iteration' , i+1)
            print(self.w)


if __name__ == "__main__":
    AND = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
    expected = np.array([0, 0, 0, 1])
    print('AND')
    perceptronAND = Perceptron(AND, expected)

    OR = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
    expected = np.array([0, 1, 1, 1])
    print('OR')
    perceptronOR = Perceptron(OR, expected)
