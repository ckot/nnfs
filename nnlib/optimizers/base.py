class Optimizer:

    def __init__(self, learning_rate=None, decay=0, **kwargs):
        # all subclasses have learning rate and decay
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        # initialize additional hyperparameters subclasses pass in
        for k, v in kwargs.items():
            setattr(self, k, v)

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
