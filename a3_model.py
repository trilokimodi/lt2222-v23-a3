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

def train(embedding):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    optimizer.zero_grad()
    predicted_label = model(embedding, offsets)
    loss = criterion(predicted_label, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    total_acc += (predicted_label.argmax(1) == label).sum().item()
    total_count += label.size(0)
    if idx % log_interval == 0 and idx > 0:
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(epoch, idx, len(dataloader), total_acc/total_count))
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

class TextClassificationModel(nn.Module):
    def __init__(self, embedded, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = torch.tensor(embedded)
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
