import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal

TIME_STEPS = 180


class Data:
    def __init__(self, features, labels, onehot=True):
        self.features = features
        self.labels = labels

        # DEBUG
        # print(np.shape(self.features))

        # convert labels to one-hot
        if onehot:
            num_classes = len(np.unique(self.labels))
            temp = np.zeros((self.labels.shape[0], num_classes))
            temp[range(self.labels.shape[0]), self.labels] = 1  # Fall:01 ADL:10
            self.labels = temp

        # shuffle data
        self.shuffle()

    def shuffle(self):
        np.random.seed(7)
        indices = np.array(range(self.labels.shape[0]))  # WHY : self.labels.shape
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]
