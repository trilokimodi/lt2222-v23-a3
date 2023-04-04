import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torchtext'])
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import io
from torch import nn
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
import re
from torchtext.data.functional import to_map_style_dataset
# Whatever other imports you need

# You can implement classes and helper functions here too.

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (text, label, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | accuracy {:8.3f}'.format(epoch, total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
         text_list.append(_text)
         label_list.append(_label)
         offsets.append(_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return text_list.to(device), label_list.to(device), offsets.to(device)

class TextDataset(Dataset):
    def __init__(self, embeddedDF=None):
          self.embeddedDF = embeddedDF

    def __len__(self):
          return len(self.embeddedDF)

    def __getitem__(self, idx):
          textTensor = self.embeddedDF.drop('class', axis=1).to_numpy()
          textTensor = torch.tensor(textTensor, dtype=torch.int64)
          label = self.embeddedDF['class']
          return textTensor, label

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size=len(vocabObject), embed_dim=50, num_class=len(labelEncoder.classes_)):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    df = pd.read_csv(args.featurefile)
    train_df = df[df['type'] == 'train']
    valid_df = df[df['type'] == 'valid']
    test_df = df[df['type'] == 'test']

    # implement everything you need here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassificationModel().to(device)

    # Hyperparameters
    EPOCHS = 10 # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64 # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    train_dff = TextDataset(train_df)
    valid_dff = TextDataset(valid_df)
    test_dff = TextDataset(test_df)

    train_dataloader = DataLoader(train_dff, batch_size=len(train_df), shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(valid_dff, batch_size=len(valid_df), shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dff, batch_size=len(test_df), shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
          scheduler.step()
        else:
          total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val))
        print('-' * 59)


    print('Checking the results of test dataset.')
    accu_test = evaluate(test_dataloader)
    print('test accuracy {:8.3f}'.format(accu_test))
