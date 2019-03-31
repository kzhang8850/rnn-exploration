import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
This network is meant to demonstrate the use of Embeddings, a new mechanism I discovered
which can help with encoding characters/words into a smaller dimensional space. It works
on the same group of words as medium_char
"""

class Trainer(object):
    """
    Trainer Pre-processing Class for data handling for the network
    """

    def __init__(self):
        """
        Variables for data handling and embeddings, and batching
        """

        self.train_batch_size = 32
        self.features = 26
        self.test_batch_size = 100
        self.training_set = None
        self.target_set = None
        self.dataset = None
        self.data_index = 0
        self.char_to_ix = {}


    def set_data(self, data):
        """
        Loads the data into the class and sets up the feature set for embedding later
        """

        data = data.split()
        self.dataset = data
        letters = "abcdefghijklmnopqrstuvwxyz"
        self.char_to_ix = {char: i for i, char in enumerate(letters)}


    def load_data(self, train=True):
        """
        Creates a batch of data, simplified because the network handles more of the
        preprocessing from before, just lists now, no more torch tensors
        """

        batch_size = self.train_batch_size if train else self.test_batch_size
        self.training_set = []
        self.target_set = []
        for i in range(batch_size):
            d = self.dataset[self.data_index]
            self.data_index = self.data_index + 1 if self.data_index + 1 < len(self.dataset) else 0

            train_seq = []
            target_seq = []
            train = d[:-1]
            target = d[1:]
            for ch in train:
                train_seq.append(ch)
            self.training_set.append(train_seq)
            for ch in target:
                self.target_set.append(ch)

        return self.training_set, self.target_set


class GRU(nn.Module):
    """
    The GRU network
    """

    def __init__(self, v_size, i_size, h_size):
        """
        Network is input into Embedding layer for encoding, then into GRU layer to get
        output and into Linear layer to squash into feature set size
        """

        super(GRU, self).__init__()
        self.embed = nn.Embedding(v_size, i_size)
        self.rnn = nn.GRU(input_size=i_size,
                        hidden_size=h_size,
                        num_layers=1,
                        batch_first=True)
        self.output = nn.Linear(h_size, v_size)


    def forward(self, input):
        """
        Chaining the layers together
        """

        embeds = self.embed(input)
        output, hidden = self.rnn(embeds, None)
        linearized = self.output(output)
        return linearized


    def get_embeddings(self, input):
        """
        For visualization or information purposes, get the embedding encoded values
        for the feature set
        """

        return self.embed(input)


if __name__=="__main__":
    # standardize randomization
    torch.manual_seed(0)

    # the network
    gru = GRU(26, 2, 128)
    print(gru)

    # Trainer and the same group of 10 words as before, all with prefix chi- and pre-
    trainer = Trainer()
    dummy_data = "chicken chimera chilled chickee present precise precede presort presold prewash"
    trainer.set_data(dummy_data)

    optimizer = optim.Adam(gru.parameters(), lr=.0005)
    loss_func = nn.CrossEntropyLoss()

    loss_cache = []
    embeddings_cache = []

    # Training loop
    for epoch in range(100):
        for _ in range(100):
            train_data, target_data = trainer.load_data()

            # converts the batch into an intermediate form for embedding
            train_ix = torch.tensor([[trainer.char_to_ix[ch] for ch in word] for word in train_data],
                                        dtype=torch.long)

            output = gru(train_ix)
            optimizer.zero_grad()

            # convert target into same intermediate form for comparison
            target = torch.tensor([trainer.char_to_ix[ch] for ch in target_data], dtype=torch.long)
            output = output.view(-1, 26)

            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        print 'Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy()

        loss_cache.append(loss.data.numpy())

    # Test trained network with manual input testing
    with torch.no_grad():
        input = ""
        print "Now the fun part :)"
        while input != "done":
            # look into prefix probability distribution
            output = ""
            input = raw_input("please give a prefix input: ")
            output += input
            for _ in range(1):
                train_ix = torch.tensor([[trainer.char_to_ix[ch] for ch in input]], dtype=torch.long)
                intermediate = gru(train_ix)
                predicted = torch.max(intermediate.data, 2)[1].numpy()
                print "distribution ", F.softmax(intermediate, dim=2)[0][-1]
                index = ord('a') + predicted[0][-1]
                next = chr(index)
                print "Next letter by highest probability ", next

        # get embeddings for visualization
        for letter in "abcdefghijklmnopqrstuvwxyz":
            embed_input = torch.tensor([trainer.char_to_ix[letter]], dtype=torch.long)
            embedding = gru.get_embeddings(embed_input)
            embeddings_cache.append((embedding[0][0].item(), embedding[0][1].item()))

        embeddings_x, embeddings_y = zip(*embeddings_cache)

        # Graph loss
        plt.figure(1)
        plt.plot(loss_cache, linewidth=5.0)
        plt.title('Medium-Hard Char GRU: Loss Analysis')
        plt.ylabel('Loss')
        plt.axis([0, 100, -.001, 3])
        plt.xlabel('Epochs')

        # Graph embeddings
        plt.figure(2)
        plt.scatter(embeddings_x, embeddings_y, c="green");
        for i, txt in enumerate("abcdefghijklmnopqrstuvwxyz"):
            plt.annotate(txt, (embeddings_x[i], embeddings_y[i]))
        plt.title('Medium-Hard Char GRU: Embeddings Analysis')
        plt.ylabel('Y Dimension in Embedding Space')
        plt.xlabel('X Dimension in Embedding Space')

        plt.show()
