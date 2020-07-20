"""Simple Neural Network with 3 layers"""
# Import
import numpy as np


# CONSTANT
LINE = 50 * '='


class NeuralNetwork:
    """Use a skeleton to make a useful format"""
    def __init__(self):
        # Get a random number
        np.random.seed()
        # Synaptic Weights - 3 Matrix with the values from -1 to 1 with mean as 0.
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    @staticmethod
    def sigmoid(x):
        """Sigmoid Normalizing Function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid Derivative Function"""
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """Training Function"""
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            # Error Weighted Derivative
            error = training_outputs - output
            # Adjust weights with the Sigmoid Derivative
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        """Thinking Function"""
        # Typecast input using .astype()
        inputs = inputs.astype(float)
        # Returns the output from new inputs.
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


while True:
    if __name__ == "__main__":

        #Initializing the neural network as a function
        neural_network = NeuralNetwork()

        print(LINE)
        print(f'Random synaptic weights:\n{neural_network.synaptic_weights}')

        # Training array
        training_inputs = np.array([[0, 0, 1],  # Should give us 0
                                    [1, 1, 1],  # Should give us 1
                                    [1, 0, 1],  # Should give us 1
                                    [0, 1, 1]])  # Should give us 0

        # Training Output
        training_outputs = np.array([[0, 1, 1, 0]]).T  # Make the array fit

        # Give the training function the input, output and iterations
        neural_network.train(training_inputs, training_outputs, 50000)  # Amount of iterations
        print(LINE + '')

        print(f'Synaptic weights after training:\n{neural_network.synaptic_weights}')
        print(LINE)
        # Ask for new values
        print('Input new values x_: 0 or 1.')
        A = str(input('Input x1: '))
        B = str(input('Input x2: '))
        C = str(input('Input x3: '))

        print(LINE)

        print(f'New situation: Input Data = {A} {B} {C}')
        print(f'Output data:\n{neural_network.think(np.array([A, B, C]))}')
        print(LINE)
        input('<Any Key to continue>')
