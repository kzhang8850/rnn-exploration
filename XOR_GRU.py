import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Trainer(object):

    def __init__(self):
        self.batch_size = 32
        self.training_set = None
        self.truth_set = None

    def load_training_data(self):
        self.training_set = np.empty([32, 2])
        self.truth_set = np.empty([32, 1])
        for _ in range(self.batch_size):
            first = np.random.choice([0, 1])
            second = np.random.choice([0, 1])
            truth = first^second
            self.training_set[i, 0] = first
            self.training_set[i, 1] = second
            self.truth_set[i, 0] = truth

        train = torch.from_numpy(self.training_set)
        truth = torch.from_numpy(self.truth_set)

        return train, truth

    def load_testing_data(self):
        self.training_set = np.empty([100, 2])
        self.truth_set = np.empty([100, 1])
        for _ in range(100):
            first = np.random.choice([0, 1])
            second = np.random.choice([0, 1])
            truth = first^second
            self.training_set[i, 0] = first
            self.training_set[i, 1] = second
            self.truth_set[i, 0] = truth

        train = torch.from_numpy(self.training_set)

        return train, self.truth_set

class GRU(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.GRU(input_size=1,
                        hidden_size=64,
                        num_layers=1,
                        batch_first=True)
        self.output = nn.Linear(64, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output, hidden = self.rnn(input, None)
        linearized = self.output(output.view(output.size(0)*output.size(1),output.size(2)))
        soft = self.softmax(linearized)
        return soft.view(output.size(0), output.size(1), output.size(2))



if __name__ == "__main__":

    gru = GRU()
    print(gru)

    trainer = Trainer()

    optimizer = optim.Adam(rnn.parameters(), lr=.005)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(100):
        for _ in range(100):
            training_set, truth = trainer.load_training_data()

            train = training_set.view(2, -1, 1)
            output = gru(train)
            optimizer.zero_grad()
            loss = loss_func(output, truth)
            optimizer.step()


        testing_set, truth = trainer.load_testing_data()
        test = testing_set.view(2, -1, 1)
        test_output = rnn(test)                   # (samples, time_step, input_size)
        pred_y = torch.max(test_output, 1)[1].numpy()
        accuracy = float((pred_y == truth).astype(int).sum()) / float(truth.size)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
