import numpy as np

class DataLoader:
    def __init__(self, dataset, *, batch_size=1, shuffle=False, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.keys = np.arange(0, len(self.dataset))
        if shuffle:
            np.random.shuffle(self.keys)
        self.num_batches = len(dataset) // batch_size
        if not drop_last and self.num_batches * batch_size < len(self.dataset):
            self.num_batches += 1

    def __iter__(self):
        """initializes the iterator, return self as an iterator object"""
        self.batch_num = 0
        return self

    def __next__(self):
        """method which actually performs the iteration"""
        # keep iterator from return nothing forever when it runs out
        # of data
        if self.batch_num >= self.num_batches:
            raise StopIteration
        # initialize lists which we'll append values to and convert
        # to ndarrays later
        X = []
        y = []
        keys = self.keys[self.batch_num * self.batch_size:
                         (self.batch_num + 1) * self.batch_size]
        # probably want to remove tqdm, using it currently to both
        # see what's going on, in particular whether drop_last works
        # as intended
        # for key in tqdm(keys):
        for key in keys:
            feats, label = self.dataset[key]
            X.append(feats)
            y.append(label)
        self.batch_num += 1
        return np.array(X), np.array(y)
