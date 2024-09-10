import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_real_path: str, data_fake_path, nrows: int=100):
        self.data_real = pd.read_csv(data_real_path, nrows=nrows)
        self.data_fake = pd.read_csv(data_fake_path, nrows=nrows)
        self.data = pd.concat([self.data_real, self.data_fake], ignore_index=True)

        self.stopwords = set(stopwords.words('english'))
   
    def init_data(self, columns: list[str]):
        self.features_raw = self.data[columns[0]].to_numpy()
        for i in range(len(self.features_raw)):
            self.features_raw[i] = re.split(r'\s+', self.features_raw[i])

        labels_real = np.ones(len(self.data_real))
        labels_fake = np.zeros(len(self.data_fake))
        self.labels = np.concatenate((labels_real, labels_fake))
    
    def get_vocab_size(self):
        return len(self.all_tokens)
    
    @staticmethod
    def _remove_stop_words(words: list[str], stopwords: set[str]) -> str:
        result = []
        for word in words:
            if word not in stopwords:
                result.append(word)
        
        return " ".join(result)
    
    @staticmethod
    def _to_lower(words: list[str]) -> list[str]:
        return [word.lower() for word in words]
    
    @staticmethod
    def _pad_sequence(sequence, max_length):
        return np.pad(sequence, (0, max_length - len(sequence)), 'constant')

    def _tokenize(self):

        self.all_tokens = set()
        tokenized_1 = []
        for i in range(len(self.features_raw)):
            tokenized_1.append(word_tokenize(self.features_raw[i]))
            for token in tokenized_1[i]:
                self.all_tokens.add(token)
        
        self.tokenizer = Tokenizer(num_words=len(self.all_tokens))    
        self.tokenizer.fit_on_texts(self.all_tokens)
        self.features = []
        max_len = 0
        for i in range(len(tokenized_1)):
            tokenized_row = self.tokenizer.texts_to_sequences(tokenized_1[i])
            cur_row = []
            for token in tokenized_row:
                if isinstance(token, list):
                    cur_row.extend(token)
                else:
                    cur_row.append(token)
            
            max_len = max(max_len, len(cur_row))
            self.features.append(cur_row)
    
        max_len *= 2
        for i in range(len(self.features)):
            self.features[i] = DataLoader._pad_sequence(self.features[i], max_len)

        self.max_len = max_len
        self.flattened_features = np.array(self.features)
    
    @staticmethod
    def tokenize_with_existing_tokenizer(features: str, tokenizer: Tokenizer, max_len: int, stopwords):
        split_features = re.split(r'\s+', features)
        lower = DataLoader._to_lower(split_features)
        without_stop_words = DataLoader._remove_stop_words(lower, stopwords)
        tokenized_1 = word_tokenize(without_stop_words)
        tokenized_row = tokenizer.texts_to_sequences(tokenized_1)
        result = []
        for token in tokenized_row:
            if isinstance(token, list):
                result.extend(token)
            else:
                result.append(token)
    
        return DataLoader._pad_sequence(result, max_len)

    
    def preprocess_data(self):
        for i in range(len(self.features_raw)):
            self.features_raw[i] = DataLoader._to_lower(self.features_raw[i])
            self.features_raw[i] = DataLoader._remove_stop_words(self.features_raw[i], self.stopwords)
        
        self._tokenize()
    
    def get(self, shuffle=False, split=[0.8, 0.1, 0.1]):
        if shuffle:
            indices = np.arange(len(self.features))
            np.random.shuffle(indices)
            self.features = self.flattened_features[indices]
            self.labels = self.labels[indices]
        
        split_1 = int(len(self.features) * split[0])
        split_2 = int(len(self.features) * (split[0] + split[1]))

        return self.features[:split_1], self.labels[:split_1], self.max_len, self.features[split_1:split_2], self.labels[split_1:split_2], self.features[split_2:], self.labels[split_2:]