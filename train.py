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
parser.add_argument('--cell_type', type=str, default='gru', help='choices: [lstm, gru, rnn]')
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
OUTPUT_SIZE = len(tgt_vocab_train)

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

enc1 = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, CELL_TYPE)
dec1 = Decoder(DEC_EMB_DIM, HID_DIM, OUTPUT_SIZE, N_LAYERS, DEC_DROPOUT, CELL_TYPE)
model = Seq2Seq(enc1, dec1).to(device)

# define the optimizer and loss function
# optimizer = optim.Adam(model.parameters())
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

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


# define a function to perform one epoch of training
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    decoded_batch_list = []
    print("THIS IS TRAINING LOOP")
    # use tqdm to show the progress bar    
    for i, batch in enumerate(tqdm(iterator)):
        src = batch[0]
        trg = batch[1]
        # print("trg: ", trg.shape)
        # swap the axes to match the input format
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
        optimizer.zero_grad()
        output = model(src, trg, args.teacher_forcing_ratio)
        output = output.permute(1, 2, 0)
        # print("output: ", output.shape)
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        # output = output[1:].view(-1, output_dim)
        # trg = trg[1:].reshape(-1)
        # print("trg: ", trg.shape)
        # loss = criterion(output, trg)
        # print("output.shape during loss: ", output.shape)
        # print("trg.shape during loss: ", trg.shape)
        # loss = F.nll_loss(output[:,:].T, trg, ignore_index=0)
        loss, preds = compute_loss(output, trg)
        decoded_batch = model.decode(src, trg, method='grid-search')
        decoded_batch_list.append(decoded_batch)
        loss = Variable(loss, requires_grad = True)
        # acc = calculate_accuracy(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        # print("output.shape before accuracy: ", output.shape)
        # print("trg.shape before accuracy: ", trg.shape) 
        # acc = calculate_char_accuracy(output[1:].view(-1, output_dim), trg[1:].reshape(-1))
        acc = calculate_accuracy(preds, trg)
        epoch_acc += acc
    # print("print sample from first decode batch")
    # print("Shape of decoded_batch_list[0]: ", decoded_batch_list[0].shape)
    # for word_index in decoded_batch_list[0]:
    #     print("word_index: ", word_index)
    #     decode_text = []
    #     for i in word_index:
    #         idx = int(i.item())
    #         decode_text.append(tgt_idx_to_word_train[idx])
    #     # decode_text = [tgt_idx_to_word_valid[i] for i in int(word_index.item())]
    #     decode_word = ' '.join(decode_text)
    #     print("pred word: ", decode_word)
    print("THIS IS TRAINING LOOP END")
    print("\n\n")
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# define a function to evaluate the model on a validation set
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    decoded_batch_list = []
    print("THIS IS EVALUATION LOOP")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch[0]
            trg = batch[1]
            # swap the axes to match the input format
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)
            output = model(src, trg, 0.0)
            output = output.permute(1, 2, 0)
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            # output = output[1:].view(-1, output_dim)
            # trg = trg[1:].reshape(-1)
            # loss = criterion(output, trg)
            # loss = F.nll_loss(output[:,:].T, trg, ignore_index=0)
            loss, preds = compute_loss(output, trg)
            # decoded_batch = model.decode(src, trg, method='grid-search')
            # decoded_batch_list.append(decoded_batch)
            # acc = calculate_accuracy(output, trg)
            # acc = calculate_char_accuracy(output[1:].view(-1, output_dim), trg[1:].reshape(-1))
            acc = calculate_accuracy(preds, trg)
            epoch_loss += loss.item()
            epoch_acc += acc
        # print("print sample from first decode batch")
        # print("Shape of decoded_batch_list[0]: ", decoded_batch_list[0].shape)
        # for word_index in decoded_batch_list[0]:
        #     print("word_index: ", word_index)
        #     decode_text = []
        #     for i in word_index:
        #         idx = int(i.item())
        #         decode_text.append(tgt_idx_to_word_train[idx])
        #     # decode_text = [tgt_idx_to_word_valid[i] for i in int(word_index.item())]
        #     decode_word = ' '.join(decode_text)
        #     print("pred word: ", decode_word)

    print("THIS IS EVALUATION LOOP END")
    print("\n\n")
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# train the model
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')
    print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_accuracy*100:.2f}')

    # print train loss and accuracy
    # print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}')

def print_execution_time():
    end_time = time.time()
    print("Execution time: ", (end_time - start_time) / 60, " minutes")

atexit.register(print_execution_time)
