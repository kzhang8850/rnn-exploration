import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
The final version of the Character Level GRU network, attempting to take on
an entire dictionary of the 10,000 most common words in the English language
"""

class Trainer(object):
    """
    Trainer Pre-processing Class for data handling for the network
    """

    def __init__(self):
        """
        Variables for data handling and embeddings, and batching
        """

        self.train_batch_size = 100
        self.features = None
        self.test_batch_size = 9970
        self.training_set = None
        self.target_set = None
        self.dataset = None
        self.data_index = 0
        self.char_to_ix = {}


    def set_data(self, data):
        """
        Loads the data into the class and sets up the feature set for embedding later
        """

        data = data.split("\n")
        self.dataset = data[:-1]

        letters = "abcdefghijklmnopqrstuvwxyz"
        self.features = len(letters)
        self.char_to_ix = {char: i for i, char in enumerate(letters)}
        self.char_to_ix["<PAD>"] = 26


    def prepare_training(self, index):
        """
        We'll be training on all the data for an adequate benchmark, this shuffles
        the data for a little bit of randomization
        """

        np.random.shuffle(self.dataset)
        self.data_index = 0


    def load_data(self, train=True):
        """
        Loads a batch of data, includes padding on the data for proper training
        later in the network, this begins the mechanism for dealing with variable length
        """

        batch_size = self.train_batch_size if train else self.test_batch_size
        batch = []
        for i in range(batch_size):
            batch.append(self.dataset[self.data_index])
            self.data_index = self.data_index + 1 if self.data_index + 1 < len(self.dataset) else 0
        batch.sort(key=len, reverse=True)
        batch = np.array(batch)

        seq_lens = []
        longest = len(batch[0])-1
        self.training_set = []
        self.target_set = []
        for b in batch:
            seq_lens.append(len(b)-1)
            train_seq = []
            target_seq = []
            train = b[:-1]
            target = b[1:]
            for ch in train:
                train_seq.append(ch)
            for j in range(longest-len(train)):
                train_seq.append("<PAD>")
            self.training_set.append(train_seq)
            for ch in target:
                self.target_set.append(ch)
            for j in range(longest-len(target)):
                self.target_set.append("<PAD>")

        return self.training_set, seq_lens, self.target_set


class GRU(nn.Module):
    """
    The GRU network
    """

    def __init__(self, v_size, i_size, l_size, h_size, o_size, pad_idx):
        """
        Network is input into Embedding layer for encoding, then into GRU layer to get
        output and into Linear layer to squash into feature set size
        Includes dropout and potentially 1+ layers
        """

        super(GRU, self).__init__()
        self.embed = nn.Embedding(num_embeddings=v_size,
                                embedding_dim=i_size,
                                padding_idx=pad_idx)
        self.rnn = nn.GRU(input_size=i_size,
                        hidden_size=h_size,
                        num_layers=l_size,
                        batch_first=True,
                        dropout=0.5 if l_size > 1 else 0)
        self.output = nn.Linear(h_size, o_size)


    def init_weights(self, m):
        """
        A weight initializer that uses orthogonal weights, potentially
        helping with faster conversion
        """

        if type(m) == nn.GRU:
            torch.nn.init.orthogonal_(m.weight_hh_l0)
            torch.nn.init.orthogonal_(m.weight_ih_l0)
            torch.nn.init.normal_(m.bias_hh_l0)
            torch.nn.init.normal_(m.bias_ih_l0)
        if type(m) == nn.Linear:
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.normal_(m.bias)


    def forward(self, input, input_lens):
        """
        Chains the layers together, first packs a padded sequence that was preprocessed,
        trains on it, then gets the padded sequences out of the pack. This takes care
        of variable length
        """

        embeds = self.embed(input)
        new_input = nn.utils.rnn.pack_padded_sequence(embeds, input_lens, batch_first=True)
        output, hidden = self.rnn(new_input, None)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(output, padding_value=0, batch_first=True)
        linearized = self.output(outputs)

        return linearized


    def get_embeddings(self, input):
        """
        For visualization or information purposes, get the embedding encoded values
        for the feature set
        """

        return self.embed(input)


def f(x, y, cache):
    return cache[(x, y)]


if __name__=="__main__":
    # standardize randomization
    torch.manual_seed(0)

    # Trainer and the dataset, which is a real dictionary of 10,000 this time
    trainer = Trainer()
    with open('words.txt') as f:
        data = f.read()
    trainer.set_data(data)

    # hyper parameter sets for training
    hyper_layers = [1, 2, 3, 4]
    hyper_hidden_units = [32, 64, 128, 256, 512, 1024]

    overall_loss_cache = {}
    embeddings_cache = []

    # Training loop

    # For all the combinations in the hyper parameter set
    for num_layers in hyper_layers:
        for num_hidden in hyper_hidden_units:
            # initializes the GRU with the hyper parameters
            gru = GRU(trainer.features+1, 2, num_layers, num_hidden, trainer.features, trainer.char_to_ix["<PAD>"])
            gru.apply(gru.init_weights)
            print gru

            optimizer = optim.Adam(gru.parameters())
            loss_func = nn.CrossEntropyLoss(ignore_index=trainer.char_to_ix["<PAD>"])

            # resets the Trainer and the data
            trainer.prepare_training(-1)
            loss_cache = []

            # trains on the data 1024 times per hyper parameter combo
            for epoch in range(1024):
                train_data, train_lengths, target_data = trainer.load_data(train=False)

                # converts the batch into an intermediate form for embedding
                train_ix = torch.tensor([[trainer.char_to_ix[ch] for ch in word] for word in train_data],
                                            dtype=torch.long)

                output = gru(train_ix, train_lengths)
                optimizer.zero_grad()

                # convert target into same intermediate form for comparison
                target = torch.tensor([trainer.char_to_ix[ch] for ch in target_data], dtype=torch.long)
                output = output.view(-1, trainer.features)

                loss = loss_func(output, target)
                loss_cache.append(loss.data.numpy())
                print 'Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy()
                loss_cache.append(loss.data.numpy())

                loss.backward()
                optimizer.step()

            overall_loss_cache[(num_layers, num_hidden)] = loss_cache

    # prints out the summary after all of the trained combos
    print "FINAL SUMMARY AFTER ALL TRAINING"
    for combo, loss in overall_loss_cache.items():
        layers, hidden = combo
        print "LAYERS:", layers, "HIDDEN SIZE:", hidden, " - ", "FINAL LOSS:", loss[-1]

    print "TRAINING FINISHED."

    # Plotting each layer, looking at the hidden unit outputs per layer
    plt.figure(1)
    plt.subplot(221)
    layer_1_loss = []
    for hidden in [32, 64, 128, 256, 512, 1024]:
        layer_1_loss.append(overall_loss_cache[(1, hidden)][-1])
    plt.plot(layer_1_loss, linewidth=5.0)
    plt.title('GRU 1 Layer Analysis')
    plt.ylabel('Loss')
    plt.axis([0, 6, -.001, 5])
    plt.subplot(222)
    layer_2_loss = []
    for hidden in [32, 64, 128, 256, 512, 1024]:
        layer_2_loss.append(overall_loss_cache[(2, hidden)][-1])
    plt.plot(layer_2_loss, color='g', linewidth=5.0)
    plt.title('GRU 2 Layer Analysis')
    plt.ylabel('Loss')
    plt.axis([0, 6, -.001, 5])
    plt.subplot(223)
    layer_3_loss = []
    for hidden in [32, 64, 128, 256, 512, 1024]:
        layer_3_loss.append(overall_loss_cache[(3, hidden)][-1])
    plt.plot(layer_3_loss, color='r', linewidth=5.0)
    plt.title('GRU 3 Layer Analysis')
    plt.ylabel('Loss')
    plt.xlabel('Hidden Size (2^(n+5)')
    plt.axis([0, 6, -.001, 5])
    plt.subplot(224)
    layer_4_loss = []
    for hidden in [32, 64, 128, 256, 512, 1024]:
        layer_4_loss.append(overall_loss_cache[(4, hidden)][-1])
    plt.plot(layer_4_loss, color='m', linewidth=5.0)
    plt.title('GRU 4 Layer Analysis')
    plt.ylabel('Loss')
    plt.xlabel('Hidden Size (2^(n+5))')
    plt.axis([0, 6, -.001, 5])

    # Plotting a 3D surface that shows correlation between layer and hidden size
    plt.figure(2)
    x = [1, 2, 3, 4]
    y = [32, 64, 128, 256, 512, 1024]
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y, overall_loss_cache)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('GRU Layer x Hidden Analysis')
    ax.set_xlabel('layers')
    ax.set_ylabel('hidden size')
    ax.set_zlabel('loss')

    plt.show()
