import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, cell_type="LSTM"):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        self.embedding = nn.Embedding(input_size, embedding_size)
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(embedding)
            # outputs shape: (seq_length, N, hidden_size)
        else:   
            outputs, hidden = self.rnn(embedding)
            cell = None

        # if self.bidrectional:
        #     hidden = torch.cat((hidden[0:self.num_layers], hidden[self.num_layers:]), dim=2)
        #     if self.cell_type == "LSTM":
        #         cell = torch.cat((cell[0:self.num_layers], cell[self.num_layers:]), dim=2)
        #     else:
        #         cell = None

        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p, cell_type="LSTM", bidirectional=True
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidrectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_size)
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p)

        # if self.cell_type == "LSTM":
        #     self.rnn = nn.LSTM(hidden_size * (1 + self.bidrectional), hidden_size, num_layers, dropout=p)
        # elif self.cell_type == "GRU":
        #     self.rnn = nn.GRU(hidden_size * (1 + self.bidrectional), hidden_size, num_layers, dropout=p)
        # else:
        #     self.rnn = nn.RNN(hidden_size * (1 + self.bidrectional), hidden_size, num_layers, dropout=p)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)


        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
            # outputs shape: (1, N, hidden_size)
        else:
            outputs, hidden = self.rnn(embedding, hidden)
            cell = None

        predictions = self.fc(outputs)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

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

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

        # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    

if __name__ == '__main__':
    # test the model
    hidden_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(input_size=100, embedding_size=10, hidden_size=hidden_size, num_layers=2, p=0.5, cell_type="RNN")
    decoder = Decoder(input_size=100, embedding_size=10, hidden_size=hidden_size, output_size=100, num_layers=2, p=0.5, cell_type="RNN")
    model = Seq2Seq(encoder, decoder).to(device)
    x = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]).to(device)
    y = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]).to(device)
    print(model(x, y).shape)
    print(model(x, y))
    