import os.path as ospath

from preprocessing import DataLoader

data_path = "../data"
fake_news_path = ospath.join(data_path, "Fake.csv")
real_news_path = ospath.join(data_path, "True.csv")

train_dataloader = DataLoader(fake_news_path, nrows=1)
train_dataloader.init_data(['title', 'text', 'subject'], 1)

features, labels = train_dataloader.get()
print(len(features[0][1]))

train_dataloader.preprocess_data()
print()
print()

features, labels = train_dataloader.get()
print(len(features[0][1]))

# print(features[:5])