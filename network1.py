import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=2, p=0.5, cell_type="LSTM", bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_size)
        if self.bidirectional:
            if self.cell_type == "LSTM":
                self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=True)
            elif self.cell_type == "GRU":
                self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=True)
            else:
                self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=True)
        else:
            if self.cell_type == "LSTM":
                self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=False)
            elif self.cell_type == "GRU":
                self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=False)
            else:
                self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=False)

    def forward(self, x):
        # x shape: (seq_length, batch_size)

        embedding = self.embedding(x)
        # embedding shape: (seq_length, batch_size, embedding_size)
        if self.bidirectional:
            if self.cell_type == "LSTM":
                outputs, (hidden, cell) = self.rnn(embedding)
                # sum bidirectional outputs
                outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
                # sum bidirectional hidden
                hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:])
                # sum bidirectional cell
                cell = (cell[0:self.num_layers] + cell[self.num_layers:])
                # outputs shape: (seq_length, batch_size, hidden_size)
            else:   
                outputs, hidden = self.rnn(embedding)
                # sum bidirectional outputs
                outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
                # sum bidirectional hidden
                hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:])
                cell = None

        else:
            if self.cell_type == "LSTM":
                outputs, (hidden, cell) = self.rnn(embedding)
                # outputs shape: (seq_length, N, hidden_size)
            else:   
                outputs, hidden = self.rnn(embedding)
                cell = None

        # outputs shape: (seq_length, batch_size, hidden_size)
        return outputs, hidden, cell
    
class Decoder(nn.Module):
    def __init__(
        self, embedding_size, hidden_size, output_size, num_layers=2, p=0.5, cell_type="LSTM", bidirectional=True
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p, inplace=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(output_size, embedding_size)
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell=None):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)  # (1, N) 

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
        predictions = predictions.squeeze(0) # (batch_size, trg_vocab_size) 

        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_token_id = 1
        self.eos_token_id = 2

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        encoder_outputs, hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[1]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            # x shape: (batch_size)
            # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
            # cell shape: (num_layers * num_directions, batch_size, hidden_size)
            output, hidden, cell = self.decoder(x, hidden, cell)
            # outputs shape: (batch_size, vocab_size)


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

    ############ GREEDY SEARCH ############
    def greedy_search_decoder(self, src, trg):
            '''
            :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
            :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
            :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
            :return: decoded_batch
            '''
            encoder_outputs, decoder_hidden, cell = self.encoder(src)
            seq_len, batch_size = trg.size()
            decoded_batch = torch.zeros((batch_size, seq_len))
            # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).cuda()
            decoder_input = Variable(trg.data[0, :]).to(device)  # [batch_size]
            for t in range(seq_len):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                topv, topi = decoder_output.data.topk(1)  
                topi = topi.view(-1)
                decoded_batch[:, t] = topi

                decoder_input = topi.detach().view(-1)  
            
            return decoded_batch  # [batch_size, seq_len]

    ############ BEAM SEARCH ############
    def beam_search_decoder(self, post, k=5):
        """Beam Search Decoder

        Parameters:

            post(Tensor) – the posterior of network.
            k(int) – beam size of decoder.

        Outputs:

            indices(Tensor) – a beam of index sequence.
            log_prob(Tensor) – a beam of log likelihood of sequence.

        Shape:

            post: (batch_size, seq_length, vocab_size).
            indices: (batch_size, beam_size, seq_length).
            log_prob: (batch_size, beam_size).

        Examples:

            >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
            >>> indices, log_prob = beam_search_decoder(post, 3)

        """
        post = post.permute(0, 2, 1)  # (batch_size, seq_length, vocab_size)
        post = torch.softmax(post, dim=-1)
        batch_size, seq_length, vocab_size = post.shape
        log_post = post.log()
        log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
        indices = indices.unsqueeze(-1)
        for i in range(1, seq_length):
            log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
            log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
            indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
        
        # Reshape indices to have shape (batch_size, k, seq_length)
        indices = indices.view(batch_size, k, seq_length)

        best_index = torch.argmax(log_prob, dim=1)
        # select the best sequence based on the best index
        best_indices = indices[torch.arange(batch_size), best_index, :]

        sorted_log_prob = log_prob[:, 0]  # Get the log probabilities from the top beam

        best_log_prob = sorted_log_prob[torch.arange(batch_size)]
        # best_indices = [batch_size, seq_len]
        return best_indices, best_log_prob
        

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
    




