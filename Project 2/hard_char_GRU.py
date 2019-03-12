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
        self.data_index = 0
        self.char_to_ix = {}


    def set_data(self, data):
        data = data.split()
        self.dataset = data
        # vocab = set(self.dataset)
        letters = "abcdefghijklmnopqrstuvwxyz"
        self.char_to_ix = {char: i for i, char in enumerate(letters)}
        self.char_to_ix["<PAD>"] = 26


    def load_data(self, train=True):
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

    def __init__(self, v_size, i_size, h_size, o_size, pad_idx):
        super(GRU, self).__init__()
        self.embed = nn.Embedding(num_embeddings=v_size,
                                embedding_dim=i_size,
                                padding_idx=pad_idx)
        self.rnn = nn.GRU(input_size=i_size,
                        hidden_size=h_size,
                        num_layers=1,
                        batch_first=True)
        self.output = nn.Linear(h_size, o_size)


    def forward(self, input, input_lens):
        embeds = self.embed(input)
        new_input = nn.utils.rnn.pack_padded_sequence(embeds, input_lens, batch_first=True)
        output, hidden = self.rnn(new_input, None)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(output, padding_value=0, batch_first=True)
        linearized = self.output(outputs)

        return linearized


if __name__=="__main__":

    trainer = Trainer()
    dummy_data = "autonomy automan automatic autograph automobile autotransformer autobiography autocracy autoimmune autotrophic chickadee chickenshit chickens chickaree chicks bigmouth biggie bigotry biggity biggest"
    trainer.set_data(dummy_data)

    gru = GRU(27, 2, 128, 26, trainer.char_to_ix["<PAD>"])
    print(gru)

    optimizer = optim.Adam(gru.parameters(), lr=.001)
    loss_func = nn.CrossEntropyLoss(ignore_index=trainer.char_to_ix["<PAD>"])

    loss_cache = []
    gradients_cache = []

    # Training loop
    for epoch in range(100):
        for _ in range(100):
            train_data, train_lengths, target_data = trainer.load_data()

            train_ix = torch.tensor([[trainer.char_to_ix[ch] for ch in word] for word in train_data],
                                        dtype=torch.long)

            output = gru(train_ix, train_lengths)
            optimizer.zero_grad()
            target = torch.tensor([trainer.char_to_ix[ch] for ch in target_data], dtype=torch.long)
            output = output.view(-1, 26)

            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        print 'Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy()

        loss_cache.append(loss.data.numpy())


    with torch.no_grad():
        input = ""
        print "Now the fun part :)"
        while input != "done":
            output = ""
            input = raw_input("please give a prefix input: ")
            output += input
            for _ in range(1):
                train_ix = torch.tensor([[trainer.char_to_ix[ch] for ch in input]], dtype=torch.long)
                input_lens = []
                input_lens.append(len(input))
                intermediate = gru(train_ix, input_lens)
                predicted = torch.max(intermediate.data, 2)[1].numpy()
                print "distribution ", F.softmax(intermediate, dim=2)[0][-1]
                index = ord('a') + predicted[0][-1]
                next = chr(index)
                print "Next letter by highest probability ", next



            # output = ""
            # input = raw_input("please a full word input: ")
            # output += input
            # for _ in range(4):
            #     test_set = np.empty([1, len(output), 26])
            #     for i in range(len(output)):
            #         hot = trainer.one_hot_encoding(output[i])
            #         test_set[0][i] = hot
            #     torch_input = torch.from_numpy(test_set).float()
            #     intermediate = gru(torch_input)
            #     predicted = torch.max(intermediate.data, 2)[1].numpy()
            #     index = ord('a') + predicted[0][-1]
            #     next = chr(index)
            #     output+= next
            #
            # print "The output from giving ", input, " is: ", output

            # print "The output from giving an 'h' is: ", output


    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(loss_cache, linewidth=5.0)
    # plt.title('XOR GRU: Loss Analysis')
    # plt.ylabel('Loss')
    # plt.axis([0, 100, -.001, 1])
    # plt.subplot(212)
    # plt.plot(accuracy_cache, color='g', linewidth=5.0)
    # plt.title('XOR GRU: Accuracy Analysis')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # plt.axis([0, 100, 0, 101])
    #
    # plt.figure(2)
    # bins = np.linspace(-.03, .03, 150, endpoint=False)
    # plt.hist(gradients_cache[0], bins, alpha=0.3, label='Epoch 1', zorder=0)
    # plt.hist(gradients_cache[1], bins, alpha=0.3, label='Epoch 10', zorder=-1)
    # plt.hist(gradients_cache[2], bins, alpha=0.3, label='Epoch 100', zorder=-2)
    # plt.title('XOR GRU: Gradient Analysis')
    # plt.ylabel('Frequency of Gradients per Bin')
    # plt.xlabel('Gradient Values')
    # plt.legend(loc='upper right')

    # plt.show()
