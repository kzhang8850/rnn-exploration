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


    def get_embeddings(self, input):
        return self.embed(input)


if __name__=="__main__":
    torch.manual_seed(0)

    trainer = Trainer()
    dummy_data = "autonomy automan automatic autograph automobile autotransformer autobiography autocracy autoimmune autotrophic chickadee chickenshit chickens chickaree chicks bigmouth biggie bigotry biggity biggest"
    trainer.set_data(dummy_data)

    gru = GRU(27, 2, 128, 26, trainer.char_to_ix["<PAD>"])
    print(gru)

    optimizer = optim.Adam(gru.parameters(), lr=.001)
    loss_func = nn.CrossEntropyLoss(ignore_index=trainer.char_to_ix["<PAD>"])

    loss_cache = []
    embeddings_cache = []

    # Training loop
    for epoch in range(10):
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
            if input != "done":
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

        for letter in "abcdefghijklmnopqrstuvwxyz":
            embed_input = torch.tensor([trainer.char_to_ix[letter]], dtype=torch.long)
            embedding = gru.get_embeddings(embed_input)
            embeddings_cache.append((embedding[0][0].item(), embedding[0][1].item()))

        embeddings_x, embeddings_y = zip(*embeddings_cache)

        plt.figure(1)
        plt.plot(loss_cache, linewidth=5.0)
        plt.title('Char GRU: Loss Analysis')
        plt.ylabel('Loss')
        plt.axis([0, 100, -.001, 1])
        plt.xlabel('Epochs')

        plt.figure(2)
        plt.scatter(embeddings_x, embeddings_y, c="green");
        for i, txt in enumerate("abcdefghijklmnopqrstuvwxyz"):
            plt.annotate(txt, (embeddings_x[i], embeddings_y[i]))
        plt.title('Char GRU: Embeddings Analysis')
        plt.ylabel('Y Dimension in Embedding Space')
        plt.xlabel('X Dimension in Embedding Space')

        plt.show()
