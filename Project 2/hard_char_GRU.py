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


    def set_data(self, data):
        data = data.split(" ")[:-1]
        self.dataset = data


    def load_data(self, train=True):
        batch_size = self.train_batch_size if train else self.test_batch_size
        batch = []
        for i in range(batch_size):
            batch.append(self.dataset[self.data_index])
            self.data_index = self.data_index + 1 if self.data_index + 1 < len(self.dataset) else 0
        batch.sort(key=len, reverse=True)
        batch = np.array(batch)

        seq_lens = []
        self.training_set = []
        self.target_set = []
        for b in batch:
            seq_lens.append(len(b)-1)
            train_seq = np.empty([len(b)-1, self.features])
            train = b[:-1]
            target = b[1:]
            for j in range(len(train)):
                ch = train[j]
                hot = self.one_hot_encoding(ch)
                train_seq[j] = hot
            self.training_set.append(torch.from_numpy(train_seq).float())

            for j in range(len(target)):
                ch = target[j]
                hot = self.one_hot_encoding(ch)
                self.target_set.append(np.ravel(np.nonzero(hot))[0])

        self.target_set = torch.from_numpy(np.array(self.target_set))

        return self.training_set, seq_lens, self.target_set


    def one_hot_encoding(self, character):
        one_hot = np.zeros([self.features])
        index = ord(character) - ord('a')
        one_hot[index] = 1

        return one_hot


    def one_hot_decoding(self, one_hot):
        decode = np.ravel(np.nonzero(one_hot))[0]
        index = ord('a') + decode
        character = chr(int(index))

        return character


class GRU(nn.Module):

    def __init__(self, i_size, h_size, o_size):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=i_size,
                        hidden_size=h_size,
                        num_layers=1,
                        batch_first=True)
        self.output = nn.Linear(h_size, o_size)


    def forward(self, input, input_lens):
        input = nn.utils.rnn.pad_sequence(input, batch_first=True)
        input = nn.utils.rnn.pack_padded_sequence(input, input_lens, batch_first=True)
        output, hidden = self.rnn(input, None)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(output, padding_value=-100, batch_first=True)
        #TODO the padded values being lost in linear layer, so i can compute loss below properly
        linearized = self.output(outputs)
        print outputs
        print linearized
        return outputs






if __name__=="__main__":

    gru = GRU(26, 64, 26)
    print(gru)

    trainer = Trainer()
    dummy_data = "autonomy automan automatic autograph automobile autotransformer autobiography autocracy autoimmune autotrophic chickadee chickenshit chickens chickaree chicks bigmouth biggie bigotry biggity biggest " * 100
    trainer.set_data(dummy_data)

    optimizer = optim.Adam(gru.parameters(), lr=.0005)
    loss_func = nn.CrossEntropyLoss()

    loss_cache = []
    gradients_cache = []

    # Training loop
    for epoch in range(30):
        for _ in range(100):
            train_data, train_lengths, target_data = trainer.load_data()

            output = gru(train_data, train_lengths)
            optimizer.zero_grad()

            target = target_data.long()
            output = output.view(-1, 26)
            # TODO can't compute loss properly because of padded values and the linear layer
            # output = torch.where()

            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        print 'Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy()

    #     # loss_cache.append(loss.data.numpy())
    #     # if epoch == 0 or epoch == 9 or epoch == 99:
    #     #     current_gradients = []
    #     #     for p in gru.parameters():
    #     #         current_gradients.extend(np.concatenate(p.grad.data.numpy(), axis=None))
    #     #     gradients_cache.append(current_gradients)
    #
    # with torch.no_grad():
    #     input = ""
    #     print "Now the fun part :)"
    #     while input != "done":
    #         output = ""
    #         input = raw_input("please give a prefix input: ")
    #         output += input
    #         for _ in range(1):
    #             test_set = np.empty([1, len(output), 26])
    #             for i in range(len(output)):
    #                 hot = trainer.one_hot_encoding(output[i])
    #                 test_set[0][i] = hot
    #             torch_input = torch.from_numpy(test_set).float()
    #             intermediate = gru(torch_input)
    #             predicted = torch.max(intermediate.data, 2)[1].numpy()
    #             print "distribution ", F.softmax(intermediate, dim=2)[0][-1]
    #             index = ord('a') + predicted[0][-1]
    #             next = chr(index)
    #             print "Next letter by highest probability ", next



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
