import os.path as ospath
from preprocessing import DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
import numpy as np

data_path = "../data"
fake_news_path = ospath.join(data_path, "Fake.csv")
real_news_path = ospath.join(data_path, "True.csv")

real_dataloader = DataLoader(fake_news_path, nrows=100)
real_dataloader.init_data(['title', 'text', 'subject'], 1)
real_dataloader.preprocess_data()

fake_dataloader = DataLoader(fake_news_path, nrows=100)
fake_dataloader.init_data(['title', 'text', 'subject'], 0)
fake_dataloader.preprocess_data()

real_features, real_labels, real_max_len = real_dataloader.get()
fake_features, fake_labels, fake_max_len = fake_dataloader.get()

# merge the two numpy arrays
all_features = np.concatenate((real_features, fake_features), axis=0)
all_labels = np.concatenate((real_labels, fake_labels), axis=0)

# shuffle the data
indices = np.arange(all_features.shape[0])
np.random.shuffle(indices)
all_features = all_features[indices]
all_labels = all_labels[indices]

model = Sequential()
model.add(Embedding(real_dataloader.get_vocab_size(), 32, input_length=real_max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(all_features, all_labels, epochs=100, batch_size=32)

real_prediction = model.predict(real_features)
fake_prediction = model.predict(fake_features)

print(f"Real news prediction: {real_prediction}", len(real_prediction))
print(f"Fake news prediction: {fake_prediction}", len(fake_prediction))