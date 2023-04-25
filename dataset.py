import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
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
        # self.labels = self.train_data.label.values
        return self.train_data, self.valid_data, self.test_data

    def create_vocab(self, data, min_freq=2):
        vocab = set()
        for word in data.str.split(' '): 
            vocab.add(word[0])
        word_to_index = {'<PAD>': 0, '<UNK>': 1}
        word_to_index = {word: index+2 for index, word in enumerate(vocab)}
        index_to_word = {index: word for word, index in word_to_index.items()}
        return vocab, word_to_index, index_to_word

    def preprocess_data(self, data):
        src_vocab, src_word_to_index, src_index_to_word = self.create_vocab(data['text'])
        tgt_vocab, tgt_word_to_index, tgt_index_to_word = self.create_vocab(data['label'])

        # Convert the text data to tensors
        src_tensor = self.text_to_tensor(data['text'], src_word_to_index)
        tgt_tensor = self.text_to_tensor(data['label'], tgt_word_to_index)
        return src_tensor, tgt_tensor, src_vocab, tgt_vocab, src_word_to_index, tgt_word_to_index, src_index_to_word, tgt_index_to_word

    def text_to_tensor(self, data, word_to_index):
        # Convert the text data to tensors
        src_tensor = torch.tensor([word_to_index[word] for word in data])
        return src_tensor

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
        row = self.data.iloc[index]
        src_word, tgt_word = row['text'], row['label']
        src_idx = self.src_word_to_index[src_word]
        tgt_idx = self.tgt_word_to_index[tgt_word]

        return src_idx, tgt_idx

def word_translation_iterator(data, src_vocab, tgt_vocab, src_word_to_idx, tgt_word_to_idx, batch_size=32, shuffle=True):
    dataset = WordTranslationDataset(data, src_vocab, tgt_vocab, src_word_to_idx, tgt_word_to_idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# define a train_iterator 
def iterator(data, labels, batch_size, shuffle=True):
    data_size = len(data)
    num_batches = int(np.ceil(data_size / float(batch_size)))
    if shuffle:
        indices = np.random.permutation(np.arange(data_size))
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_size)
        yield data[indices[start_index:end_index]], labels[indices[start_index:end_index]]



if __name__ == '__main__':
    dataset = CustomDataset(path)
    print(dataset.data)
    print(dataset.labels)