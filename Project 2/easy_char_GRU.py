import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
An easy version of a character-level GRU network that is meant to train on
a single word. It works quite easily, as expected.
"""

class Trainer(object):
    """
    The Data Pre-processing Class. Uses one-hot encoding to process data for the
    network
    """

    def __init__(self):
        """
        Variables for holding data and deciding batches
        """

        self.train_batch_size = 32
        self.features = 26
        self.test_batch_size = 100
        self.training_set = None
        self.target_set = None
        self.dataset = None
        self.data_index = 0


    def set_data(self, data):
        """
        Gets the data into the class
        """

        data = data.split(" ")[:-1] # removes the last empty element
        self.dataset = data


    def load_data(self, train=True):
        """
        Loads a batch of data as a torch tensor. Uses one-hot encoding
        """

        batch_size = self.train_batch_size if train else self.test_batch_size
        self.training_set = np.empty([batch_size, 4, self.features])
        self.target_set = np.empty([batch_size * 4])
        index = 0
        for i in range(batch_size):
            d = self.dataset[self.data_index]
            self.data_index = self.data_index + 1 if self.data_index + 1 < len(self.dataset) else 0

            train_seq = np.empty([len(d)-1, self.features])
            target_seq = np.empty([len(d)-1])
            train = d[:-1]
            target = d[1:]
            for j in range(len(train)):
                ch = train[j]
                hot = self.one_hot_encoding(ch)
                train_seq[j] = hot
            self.training_set[i] = train_seq
            for j in range(len(target)):
                ch = target[j]
                hot = self.one_hot_encoding(ch)
                self.target_set[index] = np.ravel(np.nonzero(hot))[0]
                index += 1

        torch_train = torch.from_numpy(self.training_set)
        torch_target = torch.from_numpy(self.target_set)
        return torch_train, torch_target


    def one_hot_encoding(self, character):
        """
        converts a char to an array of one-hot encoded vectors
        """

        one_hot = np.zeros([self.features])
        index = ord(character) - ord('a')
        one_hot[index] = 1

        return one_hot


    def one_hot_decoding(self, one_hot):
        """
        converts a one-hot encoded vector to a char
        """

        decode = np.ravel(np.nonzero(one_hot))[0]
        index = ord('a') + decode
        character = chr(int(index))

        return character


class GRU(nn.Module):
    """
    The GRU network
    """

    def __init__(self, i_size, h_size, o_size):
        """
        Network is input into GRU layer to get output and
        into Linear layer to squash into feature set size
        """

        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=i_size,
                        hidden_size=h_size,
                        num_layers=1,
                        batch_first=True)
        self.output = nn.Linear(h_size, o_size)


    def forward(self, input):
        """
        Chaining the layers together
        """

        output, hidden = self.rnn(input, None)
        linearized = self.output(output)
        return linearized


if __name__=="__main__":

    gru = GRU(26, 64, 26)
    print(gru)

    trainer = Trainer()

    # the data is just the single word hello
    dummy_data = "hello " * 100
    trainer.set_data(dummy_data)

    optimizer = optim.Adam(gru.parameters(), lr=.005)
    loss_func = nn.CrossEntropyLoss()

    loss_cache = []

    # Training loop
    for epoch in range(100):
        for _ in range(100):
            train_data, target_data = trainer.load_data()

            train = train_data.float()

            output = gru(train)
            optimizer.zero_grad()

            target = target_data.long()
            output = output.view(-1, 26)

            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        print 'Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy()

        loss_cache.append(loss.data.numpy())

    # Testing the trained network by outputs on manual input testing
    with torch.no_grad():
        input = ""
        print "Now the fun part :)"
        while input != "done":
            output = ""
            input = raw_input("please input an h: ")
            output += input
            for _ in range(4):
                test_set = np.empty([1, len(output), 26])
                for i in range(len(output)):
                    hot = trainer.one_hot_encoding(output[i])
                    test_set[0][i] = hot
                torch_input = torch.from_numpy(test_set).float()
                intermediate = gru(torch_input)
                predicted = torch.max(intermediate.data, 2)[1].numpy()
                index = ord('a') + predicted[0][-1]
                next = chr(index)
                output+= next

            print "The output from giving an 'h' is: ", output

    # Graphing loss
    plt.figure(1)
    plt.plot(loss_cache, linewidth=5.0)
    plt.title('Easy Char GRU: Loss Analysis')
    plt.ylabel('Loss')
    plt.axis([0, 100, -.001, .1])
    plt.xlabel('Epochs')

    plt.show()
