import nltk
nltk.download('stopwords')
nltk.download('punkt')
from tensorflow.keras.preprocessing.text import Tokenizer

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
        self.num_features = len(columns)
        self.labels = [label] * len(self.data)
    
    def _remove_stop_words(self, string: str) -> str:
        result = []
        for word in string.split():
            if word not in self.stopwords:
                result.append(word)
        
        return " ".join(result)
    
    def _to_lower(self, string: str) -> str:
        return string.lower()
    
    def _tokenize(self):
        self.features = []
        for i in range(len(self.features_raw)):
            cur_row = []
            for column in self.features_raw[i]:
                tokenizer_column = word_tokenize(column)
                cur_row.append(tokenizer_column)

            self.features.append(cur_row)
    
    def preprocess_data(self):
        for i in range(len(self.features_raw)):
            for j in range(self.num_features):
                self.features_raw[i][j] = self._remove_stop_words(self.features_raw[i][j])
                self.features_raw[i][j] = self._to_lower(self.features_raw[i][j])
        
        self._tokenize()
    
    
    def get(self):
        return (self.features_raw, self.labels)