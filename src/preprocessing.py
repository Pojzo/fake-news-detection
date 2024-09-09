import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_path: str, nrows: int=100, stop_words_path="../data/stop_words.txt"):
        self.data = pd.read_csv(data_path, nrows=nrows)
        self.stopwords = set(stopwords.words('english'))
   
    def init_data(self, columns: list[str], label: int):
        self.features_raw = self.data[columns].to_numpy()
        for i in range(len(self.features_raw)):
            for j in range(len(self.features_raw[i])):
                self.features_raw[i][j] = re.split(r'\s+', self.features_raw[i][j])

        self.num_features = len(columns)
        self.labels = [label] * len(self.data)
    
    def get_vocab_size(self):
        return len(self.all_tokens)
    
    def _remove_stop_words(self, words: list[str]) -> str:
        result = []
        for word in words:
            if word not in self.stopwords:
                result.append(word)
        
        return " ".join(result)
    
    def _to_lower(self, words: list[str]) -> list[str]:
        return [word.lower() for word in words]
    
    def _tokenize(self):
        self.all_tokens = set()
        tokenized_1 = []
        for i in range(len(self.features_raw)):
            cur_row = []
            for column in self.features_raw[i]:
                tokenized_column = word_tokenize(column)
                cur_row.append(tokenized_column)
                self.all_tokens.update(tokenized_column)

            tokenized_1.append(cur_row)
        
        self.tokenizer = Tokenizer(num_words=len(self.all_tokens))    
        self.tokenizer.fit_on_texts(self.all_tokens)
        self.features = []
        for i in range(len(tokenized_1)):
            cur_row = []
            for column in tokenized_1[i]:
                cur_row.append(self.tokenizer.texts_to_sequences(column))

            self.features.append(cur_row)
        
    
    def preprocess_data(self):
        for i in range(len(self.features_raw)):
            for j in range(self.num_features):
                self.features_raw[i][j] = self._to_lower(self.features_raw[i][j])
                self.features_raw[i][j] = self._remove_stop_words(self.features_raw[i][j])
        
        self._tokenize()
    
    def get(self):
        def pad_sequence(sequence, max_length):
            return np.pad(sequence, (0, max_length - len(sequence)), 'constant')

        max_len = 0
        flattened_features = []
        for row in self.features:
            cur_row = []
            for column in row:
                cur_column = []
                for token in column:
                    if isinstance(token, list):
                        cur_column.extend(token)
                    else:
                        cur_column.append(token)

                max_len = max(max_len, len(cur_column))
                cur_row.append(cur_column)
            
            flattened_features.append(cur_row)
        
        max_len *= 2
        for i in range(len(flattened_features)):
            for j in range(len(flattened_features[i])):
                flattened_features[i][j] = pad_sequence(flattened_features[i][j], max_len)

        return np.array(flattened_features), np.array(self.labels), max_len