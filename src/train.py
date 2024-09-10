import os.path as ospath
from preprocessing import DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
import numpy as np

data_path = "../data"
fake_news_path = ospath.join(data_path, "Fake.csv")
real_news_path = ospath.join(data_path, "True.csv")

dataloader = DataLoader(real_news_path, fake_news_path, nrows=10000)
dataloader.init_data(['text'])
dataloader.preprocess_data()

features_train, labels_train, input_len, features_test, labels_test, features_val, labels_val = dataloader.get()

model = Sequential()
model.add(Embedding(dataloader.get_vocab_size(), 32, input_length=input_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

# Create an EarlyStopping callback
train = True

early_stopping = EarlyStopping(
    monitor='val_accuracy',  
    patience=5,              
    verbose=1,               
    mode='max',              
    baseline=0.995,          
    restore_best_weights=True
)

# Fit the model passing the EarlyStopping callback
model.fit(
    np.array(features_train), 
    np.array(labels_train), 
    epochs=10, 
    batch_size=32, 
    validation_data=(np.array(features_val), np.array(labels_val)),
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(np.array(features_test), np.array(labels_test))
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
model.save("../data/model.keras")

import pickle

tokenizer = dataloader.tokenizer
tokenizer.max_len = input_len

saved_data = {
    "tokenizer": tokenizer,
    "stopwords": dataloader.stopwords,
    "max_len": input_len
}

with open("../data/data.pkl", "wb") as f:
    pickle.dump(saved_data, f)

# real_prediction = model.predict(real_features)
# fake_prediction = model.predict(fake_features)

# print(f"Real news prediction: {real_prediction}", len(real_prediction))
# print(f"Fake news prediction: {fake_prediction}", len(fake_prediction))