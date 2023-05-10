import torch 
import torch.nn as nn
import torch.nn.functional as F
import random 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, cell_type="LSTM", bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_size)
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=self.bidirectional)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=self.bidirectional)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, bidirectional=self.bidirectional)

        self.fc_hidden = nn.Linear(hidden_size * (1 + self.bidirectional), hidden_size)
        self.fc_cell = nn.Linear(hidden_size * (1 + self.bidirectional), hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        if self.cell_type == "LSTM":
            encoder_states, (hidden, cell) = self.rnn(embedding)
            # outputs shape: (seq_length, N, hidden_size)
        else:
            encoder_states, hidden = self.rnn(embedding)
            cell = None

        if self.bidirectional: 
            hidden = self.fc_hidden(torch.cat((hidden[0:self.num_layers], hidden[self.num_layers:]), dim=2))
            if self.cell_type == "LSTM":
                cell = self.fc_cell(torch.cat((cell[0:self.num_layers], cell[self.num_layers:]), dim=2))
            else: 
                cell = None
        else:
            hidden = self.fc_hidden(hidden)
            if self.cell_type == "LSTM":
                cell = self.fc_cell(cell)
            else:
                cell = None

        return encoder_states, hidden, cell
    

class Decoder(nn.Module): 
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p, cell_type="LSTM", bidirectional=True):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_size)
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size * (1 + self.bidirectional) + embedding_size, hidden_size, num_layers)
            # self.rnn = nn.LSTM(hidden_size + embedding_size, hidden_size, num_layers)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size * (1 + self.bidirectional) + embedding_size, hidden_size, num_layers)
            # self.rnn = nn.GRU(hidden_size + embedding_size, hidden_size, num_layers)
        else:
            self.rnn = nn.RNN(hidden_size * (1 + self.bidirectional) + embedding_size, hidden_size, num_layers)
            # self.rnn = nn.RNN(hidden_size + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * (2 + self.bidirectional), 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)
        encoder_states = encoder_states.repeat(self.num_layers, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)
        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            # outputs: (1, N, hidden_size)
        else:
            outputs, hidden = self.rnn(rnn_input, hidden)
            cell = None

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)

        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        # First input will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

if __name__ == '__main__':
    # test the model
    hidden_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(input_size=100, embedding_size=10, hidden_size=hidden_size, num_layers=1, p=0.5, cell_type="GRU", bidirectional=False)
    decoder = Decoder(input_size=100, embedding_size=10, hidden_size=hidden_size, output_size=100, num_layers=1, p=0.5, cell_type="GRU", bidirectional=False)
    model = Seq2Seq(encoder, decoder).to(device)
    x = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]).to(device)
    y = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]).to(device)
    output = model(x, y)
    print(output.shape)
    print(output)



