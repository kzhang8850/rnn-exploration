import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self):
        self.train_batch_size = 32
        self.features = 26
        self.test_batch_size = 100
        self.training_set = None
        self.target_set = None
        self.dataset = None


    def set_data(self, data):
        self.dataset = data


    def load_data(self, train=True):
        batch_size = self.train_batch_size if train else self.test_batch_size
        self.training_set = np.empty([batch_size])
        self.target_set = np.empty([batch_size])
        data = ("hello " * batch_size).split(" ")[:-1]
        for d in data:
            #TODO complete data generation
            # fix the dimensions of the sets
            # create an array of arrays of one hot encodings, it's a 3D array
            # (batch_size, seq_len, num_features)



    def one_hot_encoding(self, character):
        one_hot = np.zeros([self.features])
        index = ord(character) - (ord('a') - 1)
        one_hot[index] = 1

        return one_hot


    def one_hot_decoding(self, one_hot):
        decode = np.ravel(np.nonzero(one_hot))[0]
        index = (ord('a')-1) + decode
        character = chr(index)

        return character


class GRU(nn.Module):

    def __init__(self):
        super(GRU, self).__init__()



    def forward(self, input):

        




if __name__=="__main__":
