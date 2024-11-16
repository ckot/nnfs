import copy
import pickle

import numpy as np

from nnlib.activation import Activation_Softmax
from nnlib.enhancements import Activation_Softmax_Loss_CategoricalCrossentropy
from nnlib.layers import Layer_Input
from nnlib.loss import Loss_CategoricalCrossentropy

# Model class
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # count all the objects
        layer_count = len(self.layers)

        # initialize a list of trainable layers
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):
            if i == 0:
                # the first layer's previous layer object is the input layer
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                # All layers except for the first and the last
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                # The last layer the next object is the loss
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                # trainable layers have a 'weights' attribute
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # if output activation is Softmax and loss function is
        # Categorical Cross-Entropy, create an object of combined activation
        # and loss function for a substantial speedup in gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
                isinstance(self.loss, Loss_CategoricalCrossentropy):
           self.softmax_classifier_output = \
               Activation_Softmax_Loss_CategoricalCrossentropy()

    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass the output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from list, return its output
        return layer.output

    # Performs backward pass
    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        # First call backward method on the losa
        # this will set dinputs property that the last layer will access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Train the model
    def train(self, dataloader, *, epochs=1,
              print_every=1, validation_dataloader=None):

        # Main training loop
        for epoch in range(1, epochs + 1):
            if epoch == 1:
                print(f"epoch: 1 (slower than later epochs, as data is being loaded)")
            else:
                print(f"epoch: {epoch}")

            # reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step, (X, y) in enumerate(dataloader):
                if step == 1:
                    # Initialize accuracy object

                    # NOTE: (shuffling is now REQUIRED)
                    # now that we're using a dataloader, y isn't available
                    # until we fetch a batch. This isn't as ideal as we only
                    # get a peek at batchsize y values instead of entire y vals
                    self.accuracy.init(y)

                # Perform the forward pass
                output = self.forward(X, training=True)

                data_loss, regularization_loss = \
                    self.loss.calculate(output, y,
                                        include_regularization=True)

                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y)

                # Perform  backward pass
                self.backward(output, y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                if not step % print_every or step == dataloader.num_batches - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'accumulated epoch values: ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            if validation_dataloader is not None:
                self.evaluate(validation_dataloader)

    # Evaluates the model using passed-in dataset
    def evaluate(self, dataloader):

        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for X, y in dataloader:
            # Perform the forward pass
            output = self.forward(X, training=False)

            # Calculate the loss
            self.loss.calculate(output, y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    # Predicts on the samples
    def predict(self, X, *, batch_size=None):

        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):

            # if batch_size is not set -
            # use one step and full dataset
            if batch_size is None:
                batch_X = X

            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch predictions to the lsit of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):

        # Create the list of parameters
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        # Return the list of parameters
        return parameters

    # Updates the model with the new parameters
    def set_parameters(self, parameters):

        # Iterate over parameters and trainable layers
        # and update each layer with it's set of parameters
        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            # each layer's parameter_set is a tuple
            # ( weights(ndarray), biases(ndarray) )
            # which is why we're using '*' to split it
            # as separate params to the method
            layer.set_parameters(*parameter_set)

    # Saves the parameters to a file
    def save_parameters(self, path):
        # open file in binary write mode
        # and dump the params to a pickle file
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Loads the weights and updates a model instance with
    # them
    def load_parameters(self, path):

        # Open file in binary read mode
        # load params and update the trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Saves the model
    def save(self, path):

        # Make a deep copy of the current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer, remove inputs, outputs and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # Loads and returns a model
    @staticmethod
    def load(path):
        # open file in binary-read model, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # return the loaded model
        return model

