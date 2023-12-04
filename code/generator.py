import numpy as np

from keras.utils import Sequence

from utils import *


# train data pipeline
class Generator(Sequence):
    def __init__(self, x, y, batch_size=256):
        self.x = np.array(x)
        self.y = np.array(y)
        self.batch_size = batch_size
        
        self.order = np.arange(len(self.x))


    def __len__(self):
        return len(self.x) // self.batch_size


    def __getitem__(self, idx):
        batch_x = [self.x[i] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        batch_y = [self.y[i] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]

        for i in range(self.batch_size):
            _x = binary(batch_x[i], shrink=True, random=True)[0]
            _y = batch_y[i]

            if type(_y) is np.ndarray: # action
                _x, _y = augment(_x, _y)[:2]
            else: _x = augment(_x)[0] # length

            batch_x[i] = _x
            batch_y[i] = _y

        return np.array(batch_x), np.array(batch_y)


    def on_epoch_end(self):
        np.random.shuffle(self.order)
        self.x = self.x[self.order]
        self.y = self.y[self.order]