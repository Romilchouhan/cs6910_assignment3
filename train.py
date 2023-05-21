import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
from network1 import Encoder, Decoder, Seq2Seq
from attention import Encoder as AttentionEncoder, Decoder as AttentionDecoder, Seq2Seq as AttentionSeq2Seq
from dataset import CustomDataset, WordTranslationDataset, word_translation_iterator
from torch.autograd import Variable
import time
import atexit
from tqdm import tqdm
import warnings
import argparse 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from wordcloud import WordCloud
from colour import Color
from collections import Counter
import os
import PIL
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

start_time = time.time()

# set random seeds for reproducibility
SEED = 1234
torch.manual_seed(SEED)

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
parser.add_argument('--target_language', type=str, default='tam', help='target language')
parser.add_argument('--cell_type', type=str, default='gru', help='choices: [LSTM, GRU, RNN] all should be in caps')
parser.add_argument('--bidirectional', type=str, default='False', help='choices: [True, False]')
parser.add_argument('--beam_size', type=int, default=5, help='beam size for beam search')
parser.add_argument('--attention', type=str, default='False', help='choices: [True, False]')
parser.add_argument('--test', type=str, default='False', help='choices: [True, False]')
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

train_loader = word_translation_iterator(train_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE)
valid_loader = word_translation_iterator(valid_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE)
test_loader = word_translation_iterator(test_data, src_vocab_train, tgt_vocab_train, src_word_to_idx_train, tgt_word_to_idx_train, batch_size=BATCH_SIZE)

print("Time taken for data loading: ", time.time() - start_time)

if args.attention == 'True':
    enc1 = AttentionEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, CELL_TYPE, BIDIRECTIONAL)
    dec1 = AttentionDecoder(DEC_EMB_DIM, HID_DIM, OUTPUT_SIZE, N_LAYERS, DEC_DROPOUT, CELL_TYPE, BIDIRECTIONAL)
    model = AttentionSeq2Seq(enc1, dec1).to(device)

else:
    enc1 = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, CELL_TYPE, BIDIRECTIONAL)
    dec1 = Decoder(DEC_EMB_DIM, HID_DIM, OUTPUT_SIZE, N_LAYERS, DEC_DROPOUT, CELL_TYPE, BIDIRECTIONAL)
    model = Seq2Seq(enc1, dec1).to(device)

# define the optimizer 
optimizer = optim.Adam(model.parameters())

# define a function to calculate the accuracy of the model
def calculate_accuracy(predicted_output, trg):
    predicted_output = predicted_output.view(trg.shape[0], trg.shape[1])
    non_pad_elements = (trg != 0).nonzero()  # Find the indices of non-padding elements in the target tensor
    correct_elements = predicted_output[non_pad_elements].eq(trg[non_pad_elements]).sum().item()
    total_elements = trg.numel()
    accuracy = correct_elements / total_elements 
    return accuracy

def word_accuracy(prediction, trg):
    correct = 0
    for i in range(trg.shape[1]):
        correct += (prediction[:,i]==trg[:,i]).sum().item()
    return correct / trg.shape[1]

def compute_loss(preds, trg):
    trg_vocab_size = preds.shape[1]
    logits = preds.permute(0, 2, 1)
    logits = logits.contiguous().view(-1, trg_vocab_size)  # [batch_size * trg_len, trg_vocab_size]
    target_seq = trg.contiguous().view(-1) # [batch_size * trg_len]
    predicted_output = torch.argmax(logits, dim=1) # [batch_size * trg_len]
    cross_entropy_loss = F.cross_entropy(logits, target_seq, ignore_index=0)
    return cross_entropy_loss, predicted_output


def train_fn(model, iterator, optimizer, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_word_acc = 0
    print("THIS IS TRAINING LOOP")
    # use tqdm to show the progress bar
    for i, batch in enumerate(tqdm(iterator)):
        src = batch[0].to(device)  # [batch_size, src_len]
        trg = batch[1].to(device)  # [batch_size, trg_len] 
        # swap the axes to match the input format
        src = src.permute(1, 0)  
        trg = trg.permute(1, 0)   # [trg_len, batch_size]
        optimizer.zero_grad()

        #### WITHOUTH BEAM SEARCH ####
        output, best_guess = model(src, trg, args.teacher_forcing_ratio) 
        output = output.permute(1, 2, 0)  # [batch_size, target_vocab_size, trg_len]
        loss, preds = compute_loss(output.to(device), trg.to(device))
        best_guess = best_guess.permute(1, 0)  # [trg_len, batch_size]
        word_acc = word_accuracy(best_guess, trg[1:, :])
        epoch_word_acc += word_acc
        accuracy = calculate_accuracy(preds, trg.permute(1, 0))
        preds = torch.argmax(output, dim=1)  # Get the predicted labels by taking the argmax along the output dimension
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += accuracy

    print("Epoch Word accuracy: ", (epoch_word_acc / len(iterator)) * 100)
    print("THIS IS TRAINING LOOP END")
    print("\n\n")
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, beam_size=1):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_word_acc = 0
    print("THIS IS EVALUATION LOOP")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            # swap the axes to match the input format
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)
            output, best_guess = model(src, trg, 0.5) 
            output = output.permute(1, 2, 0)
            loss, preds = compute_loss(output.to(device), trg.to(device))
            best_guess = best_guess.permute(1, 0)  # [trg_len, batch_size]
            word_acc = word_accuracy(best_guess, trg[1:, :])
            accuracy = calculate_accuracy(preds, trg.permute(1, 0))
            epoch_word_acc += word_acc
            epoch_loss += loss.item()
            epoch_acc += accuracy
            best_indices, _ = model.beam_search_decoder(output, beam_size)
    print("Epoch Word accuracy: ", (epoch_word_acc / len(iterator)) * 100)
    print("THIS IS EVALUATION LOOP END")
    print("\n\n")
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# wandb sweep config
sweep_config = {
    "name": "aksharantar-sweep",
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128]},
        "epochs": {"values": [5, 10, 15]},
        "embedding_size": {"values": [64, 128, 256]},
        "num_layers": {"values": [1, 2]},
        "hidden_size": {"values": [256, 512]},
        "cell_type": {"values": ["LSTM", "GRU", "RNN"]},
        "bidirectional": {"values": [True,False]},
        "dropout": {"values": [0.4, 0.5, 0.6]},
        "beam_size": {"values": [3, 4]}
    }
}

# objective function for wandb sweep
def train_wb(config = sweep_config):
    wandb.init(project="A3 trial last", config=config)
    config = wandb.config
    wandb.run.name = "epoch_{}_cell_{}_n-layers_{}_hidden-size_{}_emb-size_{}_batch-size_{}_dropout_{}_bidirectional_{}_beam_{}".format(config.epochs,
                                                                                                            config.cell_type,
                                                                                                            config.num_layers,
                                                                                                            config.hidden_size,
                                                                                                            config.embedding_size,
                                                                                                            config.batch_size,
                                                                                                            config.dropout,
                                                                                                            config.bidirectional,
                                                                                                            config.beam_size)
    enc = AttentionEncoder(INPUT_DIM, config.embedding_size, config.hidden_size, config.num_layers, config.dropout, config.cell_type, config.bidirectional)
    dec = AttentionDecoder(config.embedding_size, config.hidden_size, OUTPUT_DIM, config.num_layers, config.dropout, config.cell_type, config.bidirectional)
    model = AttentionSeq2Seq(enc, dec).to(device)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config.epochs):
        train_loss, train_acc = train_fn(model, train_loader, optimizer, clip=1)
        val_loss, val_acc = evaluate(model, valid_loader, config.beam_size)
        wandb.log({ "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc*100, 
                    "val_loss": val_loss,
                    "val_acc": val_acc*100})


def test_fn(model, iterator, beam_size=1):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_word_acc = 0
    print("THIS IS TESTING LOOP")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)
            output, best_guess = model(src, trg, 0.5)
            best_guess = best_guess.permute(1, 0)
            output = output.permute(1, 2, 0)  # output = [batch_size, target_vocab_size, trg_len]
            loss, preds = compute_loss(output.to(device), trg.to(device))
            
            acc = calculate_accuracy(preds, trg.permute(1, 0))
            epoch_loss += loss.item()
            word_acc = word_accuracy(best_guess, trg[1:, :])
            epoch_acc += acc
            epoch_word_acc += word_acc
            # use beam search to decode
            best_indices, _ = model.beam_search_decoder(output, beam_size)  # best_indices = [trg_len, batch_size]
        print("Epoch Word accuracy: ", epoch_word_acc / len(iterator))
        print("THIS IS TESTING LOOP END")
        print("\n\n")
        return output, epoch_loss / len(iterator), epoch_acc / len(iterator)

def indexesFromWord(lang, word):
    print()
    ret = [1]
    if lang == 'en':
        for char in word:
            if char not in src_word_to_idx_train.keys():
                ret.append(src_word_to_idx_train['<UNK>'])
            else:
                ret.append(src_word_to_idx_train[char])
    else:
        for char in word:
            if char not in tgt_word_to_idx_train.keys():
                ret.append(tgt_word_to_idx_train['<UNK>'])
            else:
                ret.append(tgt_word_to_idx_train[char])
    return ret
    
def tensorsFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(2) # append <EOS> token
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorsFromWord('en', pair[0])
    target_tensor = tensorsFromWord('tam', pair[1])
    return (input_tensor, target_tensor)
        

# define a function that translates a batch of words in source language to target language
def predict(encoder, decoder, word):
    with torch.no_grad():
        input_tensor = tensorsFromWord('en', word[0])
        target_tensor = tensorsFromWord('tam', word[1])    
        # input_tensor, target_tensor = tensorsFromPair(word)
        max_length = max(input_tensor.size(0), target_tensor.size(0))
        if input_tensor.size(0) < max_length:
            input_tensor = torch.cat((input_tensor, torch.zeros(max_length - input_tensor.size(0), 1, dtype=torch.long, device=device)), dim=0)
        if target_tensor.size(0) < max_length:
            target_tensor = torch.cat((target_tensor, torch.zeros(max_length - target_tensor.size(0), 1, dtype=torch.long, device=device)), dim=0)
        input_tensor = input_tensor.permute(1, 0)
        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)

        decoder_input = target_tensor[:, 0]
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        decoded_words = []
        for di in range(max_length):
            # add batch dimension to decoder_hidden and decoder_cell
            decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
            # decoder_output = [trg_len, trg_vocab_size]
            prediction = torch.argmax(decoder_output, dim=1)
            topv, topi = prediction.data.topk(1)
            if topi.item() == 2:
                # if the predicted word is <EOS> token, then stop predicting
                break
            else:
                decoded_words.append(tgt_idx_to_word_train[topi.item()]) 
            decoder_input = prediction.squeeze().detach()
        return decoded_words
    
def predict_randomly(encoder, decoder, data_path, n=10):
    df = pd.read_csv(data_path, sep=',', header=None, names=['text', 'label'])
    df = df.sample(n).reset_index(drop=True)
    word_acc = 0
    for i in range(n):
        word = df['text'][i]
        target_word = df['label'][i]
        word1 = (word, target_word)
        input_tensor = tensorsFromWord('en', word)
        target_tensor = tensorsFromWord('tam', target_word)
        output_words = predict(encoder, decoder, word1)
        output_sentence = ' '.join(output_words)
        if output_sentence == target_word:
            word_acc += 1       
        print("Input word: ", word)
        print("Translated word: ", target_word)
        print("Predicted word: ", output_sentence)
        print("\n")
    word_acc = word_acc / n
    print("Word accuracy: ", word_acc)
    print("\n\n")
    return output_sentence

# get the output translation for the entire test set and save in csv file
def test_translate(encoder, decoder, data_path, save_path):
    df = pd.read_csv(data_path, sep=',', header=None, names=['text', 'label'])
    for i in range(len(df)):
        word = df['text'][i]
        target_word = df['label'][i]
        word1 = (word, target_word)
        input_tensor = tensorsFromWord('en', word)
        target_tensor = tensorsFromWord('tam', target_word)
        output_words = predict(encoder, decoder, word1)
        output_sentence = ' '.join(output_words)
        # save the input word, target word and predicted word in a csv file
        with open(save_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([word, target_word, output_sentence])
        # close the file
        f.close()
    return 


def translate(encoder, decoder, word):
    output_words = predict(encoder, decoder, word)
    output_sentence = ' '.join(output_words)
    return output_sentence

def predict_beam_search(encoder, decoder, word, max_length=50, beam_size=3):
    with torch.no_grad():
        input_tensor = tensorsFromWord('en', word)
        target_tensor = tensorsFromWord('tam', word)    
        input_tensor = input_tensor.permute(1, 0)
        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)

        decoder_input = target_tensor[:, 0]
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        decoded_words = []
        for di in range(max_length):
            # add batch dimension to decoder_hidden and decoder_cell
            decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
            # decoder_output = [trg_len, trg_vocab_size]
            beam_input = decoder_output.unsqueeze(0) # [1, trg_len, trg_vocab_size]
            beam_input = beam_input.permute(1, 2, 0)
            best_indices, _ = model.beam_search_decoder(beam_input, beam_size)
            prediction = best_indices.squeeze(0)
            topv, topi = prediction.data.topk(1)
            if topi.item() == 2:
                # if the predicted word is <EOS> token, then stop predicting
                break
            else:
                decoded_words.append(tgt_idx_to_word_train[topi.item()]) 
            decoder_input = prediction.squeeze().detach()
        return decoded_words

def predict_randomly_beam_search(encoder, decoder, data_path, beam_size):
    df = pd.read_csv(data_path, sep=',', header=None, names=['text', 'label'])
    word = df['text'].sample(1).values[0]
    target_word = df['label'].sample(1).values[0]
    output_words = predict_beam_search(encoder, decoder, word, len(target_word)+2, beam_size)
    output_sentence = ' '.join(output_words)
    print("Input word: ", word)
    print("Translated word: ", target_word)
    print("Predicted word: ", output_sentence)
    print("\n\n")
    return output_sentence

# Visualize model outputs
def get_colors(inputs, targets, preds):

    n = len(targets)
    smoother = SmoothingFunction().method2
    def get_scores(target, output, smoother):
        return sentence_bleu(list(list(target)), list(output), smoothing_function=smoother)

    red = Color("red")
    colors = list(red.range_to(Color("violet"),n))
    colors = list(map(lambda c: c.hex, colors))

    scores = []
    for i in range(n):
        scores.append(get_scores(targets[i], preds[i], smoother))

    d = dict(zip(sorted(scores), list(range(n))))
    ordered_colors = list(map(lambda x: colors[d[x]], scores))
    
    input_colors = dict(zip(inputs, ordered_colors))
    target_colors = dict(zip(targets, ordered_colors))
    pred_colors = dict(zip(preds, ordered_colors))

    return input_colors, target_colors, pred_colors

class Colorizer():
    def __init__(self, word_to_color, default_color):
       
        self.word_to_color = word_to_color
        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

def visualize_model_outputs(encoder, decoder, data_path, n=10):
    df = pd.read_csv(data_path, sep=',', header=None, names=['text', 'label'])
    df = df.sample(n).reset_index(drop=True)

    inputs = df['text'].astype(str).tolist()
    targets = df['label'].astype(str).tolist()
    preds = list(map(lambda x: translate(encoder, decoder, x), inputs))
    
    # Generate colors for each word
    input_colors, target_colors, pred_colors = get_colors(inputs, targets, preds)
    color_fn_ip = Colorizer(input_colors, "white")
    color_fn_tr = Colorizer(target_colors, "white")
    color_fn_op = Colorizer(pred_colors, "white")

    input_text = Counter(inputs)
    target_text = Counter(targets)
    pred_text = Counter(preds)
    fig, axs = plt.subplots(1,2, figsize=(30, 15))
    plt.tight_layout()
    font_path = "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf"
    wc_in = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(input_text)
    wc_out = WordCloud(font_path=font_path,width=800, height=400, background_color='black').generate_from_frequencies(target_text)
    wc_tar = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(pred_text)

    axs[0].set_title("Input words", fontsize=30)
    axs[0].imshow(wc_in.recolor(color_func=color_fn_ip))
    # axs[1].set_title("Target words", fontsize=30)
    # axs[1].imshow(wc_tar.recolor(color_func=color_fn_tr))
    axs[1].set_title("Targets", fontsize=30)
    axs[1].imshow(wc_out.recolor(color_func=color_fn_op))
    plt.show()


def print_execution_time():
    end_time = time.time()
    print("Execution time: ", (end_time - start_time) / 60, " minutes")

atexit.register(print_execution_time)

if __name__ == '__main__':
    # train the model
    if args.wandb == 'True':
        wandb.login(key="b3a089bfb32755711c3923f3e6ef67c0b0d2409b")
        sweep_id = wandb.sweep(sweep_config, project="A3 trial last")
        wandb.agent(sweep_id, train_wb, count=50)

    elif args.test == 'True':
        # define the best model 
        num_layers = 2
        hidden_size = 512 
        embedding_size = 256
        batch_size = 128
        dropout = 0.5
        bidirectional = False
        cell_type = 'LSTM'
        epochs = 3
        best_enc = Encoder(INPUT_DIM, embedding_size, hidden_size, num_layers, dropout, cell_type, bidirectional)
        best_dec = Decoder(embedding_size, hidden_size, OUTPUT_DIM, num_layers, dropout, cell_type, bidirectional)
        best_model = Seq2Seq(best_enc, best_dec).to(device)
        best_attn_enc = AttentionEncoder(INPUT_DIM, 512, 512, 2, 0.6, 'LSTM', False)
        best_attn_dec = AttentionDecoder(512, 512, OUTPUT_DIM, 2, 0.6, 'LSTM', False)
        best_attn_model = AttentionSeq2Seq(best_attn_enc, best_attn_dec).to(device)
        for epoch in range(epochs):
            output, test_loss, test_acc = test_fn(best_model, test_loader, BEAM_SIZE)
            print(f'Epoch: {epoch+1} | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}')
        predict_randomly(best_enc, best_dec, './aksharantar_sampled/tam/tam_test.csv', n=5, attention=False)
        predict_randomly(best_attn_enc, best_attn_dec, './aksharantar_sampled/tam/tam_test.csv', n=5)
        # predict_randomly_beam_search(best_enc, best_dec, './aksharantar_sampled/tam/tam_test.csv', beam_size=3)
        # visualize_model_outputs(best_enc, best_dec, './aksharantar_sampled/tam/tam_test.csv', n=10)
        # test_translate(best_enc, best_dec, './aksharantar_sampled/tam/tam_test.csv', './predictions_vanilla.csv')
        # test_translate(best_attn_enc, best_attn_dec, './aksharantar_sampled/tam/tam_test.csv', './predictions_attention.csv')

    else:    
        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train_fn(model, train_loader, optimizer, CLIP)
            valid_loss, valid_accuracy = evaluate(model, valid_loader, BEAM_SIZE)
            print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_accuracy*100:.2f}')


