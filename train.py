import torch
import torch.nn as nn
import torch.optim as optim
import random
from network import Encoder, Decoder, Seq2Seq
from dataset import CustomDataset, iterator, WordTranslationDataset, word_translation_iterator
from torch.autograd import Variable
import time
import atexit
from tqdm import tqdm

start_time = time.time()

# set random seeds for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# load the data
PATH = './aksharantar_sampled'
dataset = CustomDataset(PATH)
train_data, valid_data, test_data = dataset.init_dataset()
# create the vocabulary
src_vocab_train, src_word_to_idx_train, src_idx_to_word_train = dataset.create_vocab(train_data['text'])
tgt_vocab_train, tgt_word_to_idx_train, tgt_idx_to_word_train = dataset.create_vocab(train_data['label'])
src_vocab_valid, src_word_to_idx_valid, src_idx_to_word_valid = dataset.create_vocab(valid_data['text'])
tgt_vocab_valid, tgt_word_to_idx_valid, tgt_idx_to_word_valid = dataset.create_vocab(valid_data['label'])
src_vocab_test, src_word_to_idx_test, src_idx_to_word_test = dataset.create_vocab(test_data['text'])
tgt_vocab_test, tgt_word_to_idx_test, tgt_idx_to_word_test = dataset.create_vocab(test_data['label'])

# src_tensors_train, tgt_tensors_train, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, src_idx_to_word_train, tgt_idx_to_word_train = dataset.create_vocab(train_data)
# src_tensors_valid, tgt_tensors_valid, src_vocab_valid, tgt_vocab_valid, src_word_to_idx_valid, tgt_word_to_idx_valid, src_idx_to_word_valid, tgt_idx_to_word_valid = dataset.create_vocab(train_data)
# src_tensors_test, tgt_tensors_test, src_vocab_test, tgt_vocab_test, src_word_to_idx_test, tgt_word_to_idx_test, src_idx_to_word_test, tgt_idx_to_word_test = dataset.create_vocab(train_data)

# print the size of the source vocabulary
print("Source vocabulary size: ", len(src_vocab_train))
print("Target vocabulary size: ", len(tgt_vocab_train))

# define hyperparameters
INPUT_DIM = len(src_vocab_train) + 1
OUTPUT_DIM = len(tgt_vocab_train)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 128
N_EPOCHS = 10
CLIP = 1

train_loader = word_translation_iterator(train_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE)
valid_loader = word_translation_iterator(valid_data, src_vocab_valid, tgt_vocab_valid, src_word_to_idx_valid, tgt_word_to_idx_valid, batch_size=BATCH_SIZE)
test_loader = word_translation_iterator(test_data, src_vocab_test, tgt_vocab_test, src_word_to_idx_test, tgt_word_to_idx_test, batch_size=BATCH_SIZE)

print("Time taken for data loading: ", time.time() - start_time)

# define the model
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# define the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# define a function to calculate the accuracy of the model
def calculate_accuracy(preds, trg):
    # exclude the <sos> token
    preds = preds[1:].view(-1, preds.shape[-1])
    trg = trg[1:].view(-1)
    accuracy = (preds.argmax(dim=1) == trg).float().mean()
    return accuracy.item()

# define a function to perform one epoch of training
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    # use tqdm to show the progress bar    
    for i, batch in enumerate(tqdm(iterator)):
        src = batch[0]
        trg = batch[1]
        src = src.unsqueeze(0)
        trg = trg.unsqueeze(0)
        # print the shape and values of src and trg
        optimizer.zero_grad()
        output = model(src, trg)
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        print("The shape of output is: ", output.shape)
        print("The shape of trg is: ", trg.shape)
        loss = criterion(output, trg)
        loss = Variable(loss, requires_grad = True)
        acc = calculate_accuracy(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# define a function to evaluate the model on a validation set
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]
            src = src.unsqueeze(0)
            trg = trg.unsqueeze(0)
            # if a new word is there in val set not in vocab, then assign it to <unk>
            for i in range(src.shape[1]):
                if src[0][i] >= INPUT_DIM:
                    src[0][i] = 0
            output = model(src, trg, teacher_forcing_ratio=0.0)
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            acc = calculate_accuracy(output, trg)
            epoch_loss += loss.item()
            epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Train the model
N_EPOCHS = 1
CLIP = 1
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, CLIP)
    # valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion)
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     torch.save(model.state_dict(), 'tut3-model.pt')
    # print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_accuracy*100:.2f}')

    # print train loss and accuracy
    print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}')

def print_execution_time():
    end_time = time.time()
    print("Execution time: ", end_time - start_time)

atexit.register(print_execution_time)
