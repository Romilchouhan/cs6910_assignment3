import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    '''input_dim -> source vocab size'''
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Embedding(self.input_dim + 1, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.embedding(src)
        # embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input = [batch size]
        # hidden = [n layers, batch size, hid dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.embedding(input)
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is the probability that the true target will be fed into the decoder
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# add a attention layer to the basic seq2seq model
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1).transpose(0, 1)
        # hidden = [batch size, src len, hid dim]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        # encoder_outputs = [batch size, src len, hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]
        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        return torch.softmax(attention, dim=1)


if __name__ == '__main__':
    # test the model
    hidden_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(10, 10, hidden_size, 2, 0.5)
    decoder = Decoder(10, 10, hidden_size, 2, 0.5)
    model = Seq2Seq(encoder, decoder, device)
    input_seq = torch.LongTensor([[1,2,4,5,4,3,2,9], [1,2,4,5,4,3,2,9]])
    target_seq = torch.LongTensor([[1,2,4,5,4,3,2,9], [1,2,4,5,4,3,2,9]])
    output = model(input_seq, target_seq)
    print(output)
    print(output.shape)