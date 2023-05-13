import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
from collections import Counter

# Write code for dataset loading 
path = './aksharantar_sampled'

# Write a class for custom dataset loading for text data
class CustomDataset(Dataset):
    """Custom Dataset for loading Indian Language text data."""
    def __init__(self, path=path, source_lang='en', target_lang='hin', **kwargs):
        self.data_dir = path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.init_dataset()

    def init_dataset(self):
        self.train_dir = os.path.join(self.data_dir, self.target_lang, self.target_lang+'_train.csv')
        self.valid_dir = os.path.join(self.data_dir, self.target_lang, self.target_lang+'_valid.csv')
        self.test_dir = os.path.join(self.data_dir, self.target_lang, self.target_lang+'_test.csv')
        self.train_data = self.read_data(self.train_dir)
        self.valid_data = self.read_data(self.valid_dir)
        self.test_data = self.read_data(self.test_dir)
        self.data = self.train_data
        return self.train_data, self.valid_data, self.test_data

    def create_vocab(self, data):
        vocab = set()
        for word in data.str.split():
            # add every letter in the word to the vocab
            for letter in word[0]:
                vocab.add(letter)

        word_to_index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        # add the words to the vocab 
        for index, word in enumerate(vocab):
            word_to_index[word] = index+4

        index_to_word = {index: word for word, index in word_to_index.items()}
        # add the special tokens to the vocab
        vocab.add('<PAD>')
        vocab.add('<SOS>')
        vocab.add('<EOS>')
        vocab.add('<UNK>')
        return vocab, word_to_index, index_to_word

    def text_to_tensor(self, data, word_to_index):
        totensor = []  # list of letters to be converted to tensor
        # Convert the text data to tensors
        for word in data:
            for letter in word:
                if letter not in word_to_index:
                    letter = '<UNK>'
                totensor.append(word_to_index[letter])
        return totensor

    def read_data(self, file):
        # Read the csv file and return a dataframe
        data = pd.read_csv(file, sep=',', header=None, names=['text', 'label'])    
        data.text = data.text.str.lower()
        return data    

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    

class WordTranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_word_to_index, tgt_word_to_index):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_word_to_index = src_word_to_index
        self.tgt_word_to_index = tgt_word_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Returns all the letters in the word"""
        row = self.data.iloc[index]
        src_word, tgt_word = row['text'], row['label']
        src_tensor = [self.src_word_to_index['<SOS>']]
        src_tensor += self.text_to_tensor(src_word, self.src_word_to_index)  
        src_tensor.append(self.src_word_to_index['<EOS>'])

        tgt_tensor = [self.tgt_word_to_index['<SOS>']]
        tgt_tensor += self.text_to_tensor(tgt_word, self.tgt_word_to_index)
        tgt_tensor.append(self.tgt_word_to_index['<EOS>'])

        return torch.tensor(src_tensor), torch.tensor(tgt_tensor)
    
    def text_to_tensor(self, data, word_to_index):
        totensor = []  # list of letters to be converted to tensor
        # Convert the text data to tensors
        for word in data:
            for letter in word:
                if letter not in word_to_index:
                    letter = '<UNK>'
                totensor.append(word_to_index[letter])
        # src_tensor = torch.tensor(totensor, dtype=torch.long)
        return totensor

################# Collate fn ############################
class MyCollate: 
    '''
    class to add padding to the batches 
    collat_fn in dataloader is used for post processing on the batch
    '''
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # get all source indexed sentences of the batch 
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        max_length = max(max(len(src), len(tgt)) for src, tgt in zip(src_batch, tgt_batch))

        src_batch = pad_sequence([torch.nn.functional.pad(src, (0, max_length - len(src)), value=0) for src in src_batch], batch_first=True, padding_value=self.pad_idx)
        tgt_batch = pad_sequence([torch.nn.functional.pad(tgt, (0, max_length - len(tgt)), value=0) for tgt in tgt_batch], batch_first=True, padding_value=self.pad_idx)
        return src_batch, tgt_batch


def word_translation_iterator(data, src_vocab, tgt_vocab, src_word_to_idx, tgt_word_to_idx, batch_size=32, shuffle=True):
    dataset = WordTranslationDataset(data, src_vocab, tgt_vocab, src_word_to_idx, tgt_word_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=MyCollate(src_word_to_idx['<PAD>']), pin_memory=True)
    return loader



if __name__ == '__main__':
    dataset = CustomDataset(path)
    print(dataset.data)
    print(dataset.labels)