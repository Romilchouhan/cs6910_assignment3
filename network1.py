import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random 
from queue import PriorityQueue
import operator

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
        # x shape: (seq_length, N) where N is batch size

        embedding = self.embedding(x)
        # embedding shape: (seq_length, N, embedding_size)
        if self.bidirectional:
            if self.cell_type == "LSTM":
                outputs, (hidden, cell) = self.rnn(embedding)
                # sum bidirectional outputs
                outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
                # sum bidirectional hidden
                hidden = (hidden[0:self.num_layers] + hidden[self.num_layers:])
                # outputs shape: (seq_length, N, hidden_size)
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

        # self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding = nn.Embedding(output_size, embedding_size)
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=p)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=p)

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
        self.sos_token_id = 1
        self.eos_token_id = 2

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_outputs, hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[1]

        for t in range(1, target_len):
            # print("x shape: ", x.shape)
            # print("hidden shape: ", hidden.shape)
            # print("cell shape: ", cell.shape)
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
    
    def decode(self, src, trg, method='beam-search'):
        encoder_output, hidden, cell = self.encoder(src)
        # print("encoder_output shape in decode: ", encoder_output.shape)
        hidden = hidden[:self.decoder.num_layers]
        if method == 'beam-search':
            # return self.beam_decode(trg, hidden, encoder_output)
            # outputs = self.forward(src, trg)
            # print("src shape: ", src.shape)
            # print("outputs shape: ", outputs.shape)
            return self.beam_search_decode(src, beam_width=5)
        else:
            return self.greedy_decode(trg, hidden, encoder_output)

    ############ GREEDY SEARCH ############
    def greedy_decode(self, trg, decoder_hidden, encoder_outputs, ):
            '''
            :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
            :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
            :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
            :return: decoded_batch
            '''
            seq_len, batch_size = trg.size()
            decoded_batch = torch.zeros((batch_size, seq_len))
            # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).cuda()
            decoder_input = Variable(trg.data[0, :]).to(device)  # sos
            for t in range(seq_len):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                topv, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
                topi = topi.view(-1)
                decoded_batch[:, t] = topi

                decoder_input = topi.detach().view(-1)  
            
            return decoded_batch

    ############ BEAM SEARCH ############
    def beam_search_decode(self, src, beam_width):
        self.eval()

        max_len = src.shape[0]  # Maximum sentence length
        batch_size = src.shape[1]  # Batch size
        # Encode the source sequence
        encoder_output, hidden, cell = self.encoder(src)

        # Initialize the beam search
        beam_outputs = [[torch.tensor([1])]]  # List of completed output sequences
        beam_scores = torch.zeros(1, dtype=torch.float)  # Scores for each output sequence
        # hidden = (hidden[0].repeat(1, beam_width, 1), hidden[1].repeat(1, beam_width, 1))  # Repeat hidden state for beam width
        # print("hidden shape: ", hidden.shape)

        for _ in range(max_len):
            current_candidates = []  # List to store new candidate sequences

            # Generate candidates for each current beam
            for output in beam_outputs:
                # Prepare input for the decoder
                # trg = torch.tensor(output).unsqueeze(0)  # Add a batch dimension
                trg = torch.ones(batch_size, dtype=torch.long).to(device)  # Fill with <SOS> tokens
                # trg = trg.transpose(0, 1)  # Transpose to shape (seq_len, batch_size)
                # print("baam scores shape: ", beam_scores.shape) 
                # print("trg shape: ", trg.shape)
                # print("hidden shape: ", hidden.shape)
                # print("cell shape: ", cell.shape)

                # Perform a forward pass through the decoder
                decoder_output, hidden, _ = self.decoder(trg, hidden, cell)

                # Get the log probabilities for the next token
                log_probs = F.log_softmax(decoder_output.squeeze(0), dim=1)

                # Get the top-k candidates and their log probabilities
                topk_probs, topk_ids = log_probs.topk(beam_width, dim=1)

                # Expand the current beam
                for i in range(beam_width):
                    candidate_seq = output + [topk_ids[0][i].item()]  # Extend the sequence
                    candidate_score = beam_scores + topk_probs[0][i].item()  # Accumulate the score
                    
                    if topk_ids[0][i].item() == self.eos_token_id:  # Check if the candidate is complete
                        beam_outputs.append(candidate_seq)
                        beam_scores = torch.cat((beam_scores, candidate_score.unsqueeze(0)), dim=0)
                    else:
                        # print('candidate_seq: ', candidate_seq)
                        # print('candidate_score: ', candidate_score)
                        for c in candidate_score:
                            current_candidates.append((candidate_seq, c.item()))
                        # current_candidates.append((candidate_seq, candidate_score.item()))

            # Sort the candidates based on scores
            # print("I reached here")
            current_candidates.sort(key=lambda x: x[1], reverse=True)
            # current_candidates = sorted(current_candidates, key=lambda x: x[1], reverse=True)

            # Select top-k candidates for the next iteration
            beam_outputs = [candidate[0] for candidate in current_candidates[:beam_width]]
            beam_scores = torch.tensor([candidate[1] for candidate in current_candidates[:beam_width]])

            if all(output[-1] == self.eos_token_id for output in beam_outputs):
                break
        print("beam outputs shape: ", len(beam_outputs[0]))
        best_output = torch.tensor(beam_outputs[beam_scores.argmax().item()])

        return best_output
    
   
    

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
    