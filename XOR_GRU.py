import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
This script is a Trivial GRU Recurrent Neural Network that can solve the XOR operation
problem: that is, given the first two binary digits in a length three sequence,
figure out that the third digit is the XOR operation performed on the first two.
Includes an easy-to-use data generator for the network, and also visualizations.

by KZ w/ help from Paul Ruvolo
"""

class Trainer(object):
    """
    A generic XOR data generator
    """

    def __init__(self):
        """
        Some variables for holding the data
        """

        self.train_batch_size = 32
        self.test_batch_size = 100
        self.training_set = None
        self.truth_set = None


    def load_data(self, train=True):
        """
        generates a set of random pairs of 0s and 1s with a XORed target,
        will create a different batch size of these data depending on training
        or validating
        """

        batch_size = self.train_batch_size if train else self.test_batch_size
        self.training_set = np.empty([batch_size, 2])
        self.truth_set = np.empty([batch_size, 1])
        for i in range(batch_size):
            first = np.random.choice([0, 1])
            second = np.random.choice([0, 1])
            truth = first^second
            self.training_set[i, 0] = first
            self.training_set[i, 1] = second
            self.truth_set[i, 0] = truth

        train = torch.from_numpy(self.training_set)
        truth = torch.from_numpy(self.truth_set)

        return train, truth


class GRU(nn.Module):
    """
    My GRU Network Model
    """

    def __init__(self):
        """
        This is where you define the different pieces to the hidden layer,
        for me it's my GRU layer, which is squashed into two probabilities for 0 or 1
        by a Linear unit, and then put between 0 and 1 by Softmax.
        """

        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=1,
                        hidden_size=4,
                        num_layers=1,
                        batch_first=True)
        self.output = nn.Linear(4, 2)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input):
        """
        This is there you set up the network itself, connecting all of the pieces
        you defined in the init starting from input and then ending at the output of
        the feed forward.
        """

        output, hidden = self.rnn(input, None)
        linearized = self.output(output[:, -1, :])
        soft = self.softmax(linearized)
        return soft



if __name__ == "__main__":
    # initialize the network, and print it for good measure
    gru = GRU()
    print(gru)

    # initialize the data generator
    trainer = Trainer()

    # initialize network updaters
    optimizer = optim.Adam(gru.parameters(), lr=.005)
    loss_func = nn.CrossEntropyLoss()

    # storing data for visualization
    loss_cache = []
    accuracy_cache = []
    gradients_cache = []

    # Training loop
    for epoch in range(100):
        for _ in range(100):
            # grab a batch of data
            training_set, target = trainer.load_data()

            # reshape training set to be
            # (batch_size, time_step (seq_len), input_size (num_features))
            train = training_set.view(-1, 2, 1).float()

            # feed forward input through the network
            output = gru(train)
            optimizer.zero_grad()

            # reshape target to be just the batch size
            target = target.view(trainer.train_batch_size).long()

            # compute loss, BPTT, and update the network
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        testing_set, truth = trainer.load_data(train=False)
        test = testing_set.view(-1, 2, 1).float()

        # Send through network, this time grabbing the best value and doing
        # comparisons to calculate an accuracy metric
        test_output = gru(test)
        pred_y = torch.max(test_output, 1)[1].numpy()
        truth = truth.view(trainer.test_batch_size).numpy()
        accuracy = float((pred_y == truth).astype(int).sum()) / float(truth.size)
        print 'Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy

        # record data for visualization later
        loss_cache.append(loss.data.numpy())
        accuracy_cache.append(accuracy*100)
        if epoch == 0 or epoch == 9 or epoch == 99:
            current_gradients = []
            for p in gru.parameters():
                current_gradients.extend(np.concatenate(p.grad.data.numpy(), axis=None))
            gradients_cache.append(current_gradients)


    # Now that training is over, matplotlib visualizations
    plt.figure(1)
    plt.subplot(211)
    plt.plot(loss_cache, linewidth=5.0)
    plt.title('XOR GRU: Loss Analysis')
    plt.ylabel('Loss')
    plt.axis([0, 100, -.001, 1])
    plt.subplot(212)
    plt.plot(accuracy_cache, color='g', linewidth=5.0)
    plt.title('XOR GRU: Accuracy Analysis')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.axis([0, 100, 0, 101])

    plt.figure(2)
    bins = np.linspace(-.03, .03, 150, endpoint=False)
    plt.hist(gradients_cache[0], bins, alpha=0.3, label='Epoch 1', zorder=0)
    plt.hist(gradients_cache[1], bins, alpha=0.3, label='Epoch 10', zorder=-1)
    plt.hist(gradients_cache[2], bins, alpha=0.3, label='Epoch 100', zorder=-2)
    plt.title('XOR GRU: Gradient Analysis')
    plt.ylabel('Frequency of Gradients per Bin')
    plt.xlabel('Gradient Values')
    plt.legend(loc='upper right')

    plt.show()
