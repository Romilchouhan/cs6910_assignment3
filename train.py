import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
# from network import Encoder, Decoder, Seq2Seq
# from network1 import Encoder, Decoder, Seq2Seq
from attention import Encoder, Decoder, Seq2Seq
from dataset import CustomDataset, WordTranslationDataset, word_translation_iterator
from torch.autograd import Variable
import time
import atexit
from tqdm import tqdm
import warnings
import argparse 
warnings.filterwarnings("ignore")

start_time = time.time()

# set random seeds for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Seq2Seq Model on Aksharantar Dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--embedding_size', type=int, default=256, help='embedding size for encoder and decoder')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden size for encoder and decoder')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers for encoder and decoder')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate for encoder and decoder')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='teacher forcing ratio')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
parser.add_argument('--target_language', type=str, default='hin', help='target language')
parser.add_argument('--cell_type', type=str, default='gru', help='choices: [LSTM, GRU, RNN] all should be in caps')
parser.add_argument('--bidirectional', type=str, default='False', help='choices: [True, False]')
parser.add_argument('--beam_size', type=int, default=5, help='beam size for beam search')
args = parser.parse_args()

# load the data
PATH = './aksharantar_sampled'
dataset = CustomDataset(PATH, target_lang=args.target_language)
train_data, valid_data, test_data = dataset.init_dataset()
# create the vocabulary
src_vocab_train, src_word_to_idx_train, src_idx_to_word_train = dataset.create_vocab(train_data['text'])
tgt_vocab_train, tgt_word_to_idx_train, tgt_idx_to_word_train = dataset.create_vocab(train_data['label'])
src_vocab_valid, src_word_to_idx_valid, src_idx_to_word_valid = dataset.create_vocab(valid_data['text'])
tgt_vocab_valid, tgt_word_to_idx_valid, tgt_idx_to_word_valid = dataset.create_vocab(valid_data['label'])
src_vocab_test, src_word_to_idx_test, src_idx_to_word_test = dataset.create_vocab(test_data['text'])
tgt_vocab_test, tgt_word_to_idx_test, tgt_idx_to_word_test = dataset.create_vocab(test_data['label'])

# print the size of the source vocabulary
print("Source vocabulary size: ", len(src_vocab_train))
print("Target vocabulary size: ", len(tgt_vocab_train))

# define hyperparameters
INPUT_DIM = len(src_vocab_train)
OUTPUT_DIM = len(tgt_vocab_train)
ENC_EMB_DIM = args.embedding_size
DEC_EMB_DIM = args.embedding_size
HID_DIM = args.hidden_size
N_LAYERS = args.num_layers
ENC_DROPOUT = args.dropout
DEC_DROPOUT = args.dropout
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
CLIP = args.clip
CELL_TYPE = args.cell_type
BIDIRECTIONAL = args.bidirectional
OUTPUT_SIZE = len(tgt_vocab_train)

# convert bidirectional from str to bool
if BIDIRECTIONAL == 'True':
    BIDIRECTIONAL = True
else:
    BIDIRECTIONAL = False

print("This is the cell type: ", CELL_TYPE)

train_loader = word_translation_iterator(train_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE)
# valid_loader = word_translation_iterator(valid_data, src_vocab_valid, tgt_vocab_valid, src_word_to_idx_valid, tgt_word_to_idx_valid, batch_size=BATCH_SIZE)
valid_loader = word_translation_iterator(valid_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE)
test_loader = word_translation_iterator(test_data, src_vocab_test, tgt_vocab_test, src_word_to_idx_test, tgt_word_to_idx_test, batch_size=BATCH_SIZE)

print("Time taken for data loading: ", time.time() - start_time)

# define the model
# enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, CELL_TYPE)
# dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, CELL_TYPE)
# model = Seq2Seq(enc, dec, device).to(device)

enc1 = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, CELL_TYPE, BIDIRECTIONAL)
dec1 = Decoder(DEC_EMB_DIM, HID_DIM, OUTPUT_SIZE, N_LAYERS, DEC_DROPOUT, CELL_TYPE, BIDIRECTIONAL)
model = Seq2Seq(enc1, dec1).to(device)

# define the optimizer 
optimizer = optim.Adam(model.parameters())

# define a function to calculate the accuracy of the model
def calculate_accuracy(predicted_output, trg):
    predicted_output = predicted_output.view(trg.shape[0], trg.shape[1])
    non_pad_elements = (trg != 0).nonzero()  # Find the indices of non-padding elements in the target tensor
    correct_elements = predicted_output[non_pad_elements[:, 0]].eq(trg[non_pad_elements[:, 0]]).sum().item()
    total_elements = non_pad_elements.shape[0]
    accuracy = correct_elements / total_elements 
    return accuracy

def calculate_char_accuracy(preds, trg):
    print("preds shape: ", preds.shape)
    print("trg shape: ", trg.shape)
    preds = torch.argmax(preds, dim=1)  # Get the predicted labels by taking the argmax along the output dimension
    non_pad_elements = (trg != 0).nonzero()  # Find the indices of non-padding elements in the target tensor
    correct_chars = preds[non_pad_elements[:, 0]].eq(trg[non_pad_elements[:, 0]]).sum().item()
    total_chars = non_pad_elements.shape[0]
    print("correct chars: ", correct_chars)
    print("total chars: ", total_chars)
    char_accuracy = correct_chars / total_chars 
    return char_accuracy

def compute_loss(preds, trg):
    trg_vocab_size = preds.shape[1]
    logits = preds.permute(0, 1, 2)
    logits = logits.contiguous().view(-1, trg_vocab_size)  
    target_seq = trg.contiguous().view(-1)

    predicted_output = torch.argmax(logits, dim=1)
    cross_entropy_loss = F.cross_entropy(logits, target_seq, ignore_index=0)
    return cross_entropy_loss, predicted_output


def train(model, iterator, optimizer, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    print("THIS IS TRAINING LOOP")
    # use tqdm to show the progress bar
    for i, batch in enumerate(tqdm(iterator)):
        src = batch[0]
        trg = batch[1]
        # swap the axes to match the input format
        src = src.permute(1, 0)  # [src_len, batch_size]
        trg = trg.permute(1, 0)  # [trg_len, batch_size]
        optimizer.zero_grad()

        #### WITHOUTH BEAM SEARCH ####
        output = model(src, trg, args.teacher_forcing_ratio) 
        output = output.permute(1, 2, 0)  # [batch_size, target_vocab_size, trg_len]
        loss, preds = compute_loss(output, trg)
        accuracy = calculate_accuracy(preds, trg)
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += accuracy

    print("THIS IS TRAINING LOOP END")
    print("\n\n")
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):
    model.eval()
    epoch_acc = 0
    print("THIS IS EVALUATION LOOP")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch[0]
            trg = batch[1]
            # swap the axes to match the input format
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)
            output = model(src, trg, 0.0)
            output = output.permute(1, 2, 0)  # output = [batch_size, target_vocab_size, trg_len]
            loss, preds = compute_loss(output, trg)
            # use beam search to decode
            # best_indices, _ = model.beam_search_decoder(output)
            best_indices, _ = model.greedy_search_decoder(output)
            predicted = best_indices.to(torch.float)
            trg = trg.to(torch.float)
            acc = calculate_accuracy(predicted, trg)            
            epoch_acc += acc
    print("THIS IS EVALUATION LOOP END")
    print("\n\n")
    return epoch_acc / len(iterator)

# train the model
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, CLIP)
    valid_accuracy = evaluate(model, valid_loader)
    print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Val. Acc: {valid_accuracy*100:.2f}')


def print_execution_time():
    end_time = time.time()
    print("Execution time: ", (end_time - start_time) / 60, " minutes")

atexit.register(print_execution_time)
