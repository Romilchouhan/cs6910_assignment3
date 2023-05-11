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

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_outputs, hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[1]

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
    
    def decode(self, src, trg, method='beam-search'):
        encoder_output, hidden, cell = self.encoder(src)
        # print("encoder_output shape in decode: ", encoder_output.shape)
        hidden = hidden[:self.decoder.num_layers]
        if method == 'beam-search':
            return self.beam_decode(trg, hidden, encoder_output)
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
    def beam_decode(self, trg, decoder_hiddens, encoder_outputs=None):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is batch size and T is the maximum length of the output sequence
        :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sequence
        '''
        # target_tensor = target_tensor.permute(1, 0)
        print(trg.shape)
        beam_width = 10
        topk = 1  # how many candidate you want to generate
        decoded_batch = []

        # decoding goes letter by letter
        for idx in range(trg.size(0)):  # batch_size
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)  # [1, B, H]=>[1, 1, H]
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1) if encoder_outputs is not None else None

            print("encoder_output", encoder_output.shape)

            # Start with the start of the sentence token
            # decoder_input = torch.LongTensor([[0]]).to(device)
            decoder_input = Variable(trg.data[0, :]).to(device)  # sos (This is the change)
            print("decoder_hidden", decoder_hidden.shape)
            # Number of words to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search 
            while True: 
                # give up if decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                print("decoder_input", decoder_input.shape)
                decoder_hidden = n.h

                if n.wordid.item() == 1 and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue
                
                # decode for one step using decoder
                decoder_hidden = decoder_hidden.squeeze(0)
                # asert all the input sizes are 2D or 3D
                print("decoder_input shape", decoder_input.shape)
                print("decoder_hidden shape", decoder_hidden.shape)
                print("encoder_output shape", encoder_output.shape)
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output.squeeze(1))  

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch




class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        """
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    
    def __lt__(self, other):
        return self.leng < other.leng
    
    def __gt__(self, other):
        return self.leng > other.leng
    

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
    