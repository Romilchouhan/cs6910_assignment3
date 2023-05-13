import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
# from network import Encoder, Decoder, Seq2Seq
from network1 import Encoder, Decoder, Seq2Seq
# from attention import Encoder, Decoder, Seq2Seq
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
parser.add_argument('--wandb', type=str, default='False', help='choices: [True, False]')
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
BEAM_SIZE = args.beam_size

# convert bidirectional from str to bool
if BIDIRECTIONAL == 'True':
    BIDIRECTIONAL = True
else:
    BIDIRECTIONAL = False

print("This is the cell type: ", CELL_TYPE)

train_loader = word_translation_iterator(train_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE).to(device)
valid_loader = word_translation_iterator(valid_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE).to(device)
test_loader = word_translation_iterator(test_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE).to(device)

print("Time taken for data loading: ", time.time() - start_time)


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
    total_elements = trg.numel()
    accuracy = correct_elements / total_elements 
    return accuracy

def compute_loss(preds, trg):
    trg_vocab_size = preds.shape[1]
    logits = preds.permute(0, 1, 2)
    logits = logits.contiguous().view(-1, trg_vocab_size)  
    target_seq = trg.contiguous().view(-1)

    predicted_output = torch.argmax(logits, dim=1)
    cross_entropy_loss = F.cross_entropy(logits, target_seq, ignore_index=0)
    return cross_entropy_loss, predicted_output

def train_fn(model, iterator, optimizer, clip):
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
        preds = torch.argmax(output, dim=1)  # Get the predicted labels by taking the argmax along the output dimension
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += accuracy

    print("THIS IS TRAINING LOOP END")
    print("\n\n")
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, beam_size=1):
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
            best_indices, _ = model.beam_search_decoder(output, beam_size)
            # best_indices, _ = model.greedy_search_decoder(output)
            word_acc = calculate_accuracy(preds, trg)       
            preds = best_indices
            epoch_acc += word_acc
    print("THIS IS EVALUATION LOOP END")
    print("\n\n")
    return epoch_acc / len(iterator)

# wandb sweep config
sweep_config = {
    "name": "aksharantar-sweep",
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [64, 128]},
        "epochs": {"values": [3, 5, 10]},
        "embedding_size": {"values": [128, 256, 512]},
        "num_layers": {"values": [1, 2, 3]},
        "hidden_size": {"values": [256, 512, 1024]},
        "cell_type": {"values": ["LSTM", "GRU", "RNN"]},
        "bidirectional": {"values": ["True", "False"]},
        "dropout": {"values": [0.2, 0.3, 0.4, 0.5]},
        "beam_size": {"values": [3, 4, 5, 7]},
        "optimizer": {"values": ["Adam", "SGD"]}
    }
}

# objective function for wandb sweep
def train_wb(config = sweep_config):
    wandb.init(project="aksharantar", config=config)
    config = wandb.config
    wandb.run.name = "epoch_{}_cell_{}_n-layers_{}_hidden-size_{}_emb-size_{}_batch-size_{}_dropout_{}_bidirectional_{}_beam_{}_opt_{}".format(config.epochs,
                                                                                                            config.cell_type,
                                                                                                            config.num_layers,
                                                                                                            config.hidden_size,
                                                                                                            config.embedding_size,
                                                                                                            config.batch_size,
                                                                                                            config.dropout,
                                                                                                            config.bidirectional,
                                                                                                            config.beam_size,
                                                                                                            config.optimizer)
    enc = Encoder(INPUT_DIM, config.embedding_size, config.hidden_size, config.num_layers, config.dropout, config.cell_type, config.bidirectional)
    dec = Decoder(config.embedding_size, config.hidden_size, OUTPUT_DIM, config.num_layers, config.dropout, config.cell_type, config.bidirectional)
    model = Seq2Seq(enc, dec).to(device)

    # initialize optimizer
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_loss, train_acc = train_fn(model, train_loader, optimizer, clip=1)
    val_acc = evaluate(model, valid_loader, config.beam_size)
    wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})


def print_execution_time():
    end_time = time.time()
    print("Execution time: ", (end_time - start_time) / 60, " minutes")

atexit.register(print_execution_time)

if __name__ == '__main__':
    # train the model
    if args.wandb == 'True':
        wandb.login(key="b3a089bfb32755711c3923f3e6ef67c0b0d2409b")
        sweep_id = wandb.sweep(sweep_config, project="aksharantar")
        wandb.agent(sweep_id, train_wb, count=40)
        
    else:    
        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train_fn(model, train_loader, optimizer, CLIP)
            valid_accuracy = evaluate(model, valid_loader, BEAM_SIZE)
            print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Val. Acc: {valid_accuracy*100:.2f}')


