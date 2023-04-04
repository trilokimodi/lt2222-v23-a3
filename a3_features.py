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


def yield_tokens(paths):
    tokenizer = get_tokenizer('basic_english')
    for iPath in paths:
        with io.open(iPath, encoding = 'utf-8') as fh:
            for line in fh:
                if line.startswith(('Message-ID', 'Mime-Version:', 'Content-Type:', 'Content-Transfer-Encoding:')):
                    pass
                elif line.startswith(('X-From:', 'X-To:', 'X-cc:', 'X-bcc:', 'X-Folder:', 'X-Origin:', 'X-FileName:')):
                    pass
                elif re.search('From:', line) is not None:
                    pass
                elif re.search('To:', line) is not None:
                    pass
                elif re.search('Date:', line) is not None:
                    pass
                elif re.search('Sent:', line) is not None:
                    pass
                elif re.search(' -----Original Message-----', line) is not None:
                    pass
                else:
                    yield tokenizer(line)
        fh.close()
        
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
    def __init__(self, dataset_paths, labelEncoder, tokenizer, vocab):
        self.datasetPaths = dataset_paths
        self.labelEncoder = labelEncoder
        self.tokenizer=tokenizer
        self.vocab=vocab

    def __len__(self):
        return len(self.datasetPaths)

    def __getitem__(self, idx):
        docFilePath = self.datasetPaths[idx]
        text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        newFile = list()
        with io.open(docFilePath, encoding = 'utf-8') as fh:
            for line in fh:
                if line.startswith(('Message-ID', 'Mime-Version:', 'Content-Type:', 'Content-Transfer-Encoding:')):
                    pass
                elif line.startswith(('X-From:', 'X-To:', 'X-cc:', 'X-bcc:', 'X-Folder:', 'X-Origin:', 'X-FileName:')):
                    pass
                elif re.search('From:', line) is not None:
                    pass
                elif re.search('To:', line) is not None:
                    pass
                elif re.search('Date:', line) is not None:
                    pass
                elif re.search('Sent:', line) is not None:
                    pass
                elif re.search(' -----Original Message-----', line) is not None:
                    pass
                else:
                    newFile.append(line)
        content = ''.join(newFile)
        textTensor = torch.tensor(text_pipeline(content), dtype=torch.int64)
        label = docFilePath.split('/')[-2]
        label = self.labelEncoder.transform([label])
        return textTensor, label[0]
    
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
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
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    dataset = args.inputdir
    dataset_paths = [os.path.join(os.path.join(dataset, iClass), iText) for iClass in os.listdir(dataset) for iText in os.listdir(os.path.join(dataset, iClass))]

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    tokenizer = get_tokenizer('basic_english')
    labelEncoder = LabelEncoder()
    labelEncoder.fit(os.listdir(dataset))
    
    vocabObject = build_vocab_from_iterator(yield_tokens(dataset_paths), specials=["<unk>"])
    vocabObject.set_default_index(vocabObject["<unk>"])
    
    complete_data = TextDataset(dataset_paths=dataset_paths, labelEncoder=labelEncoder, tokenizer=tokenizer, vocab=vocabObject)
    train_prop = int(len(complete_data) * (int(args.testsize)/100))
    train_data, test_data = random_split(complete_data, [train_prop, len(complete_data) - train_prop])
    train_dataset = to_map_style_dataset(train_data)
    test_dataset = to_map_style_dataset(test_data)
    num_train = int(len(train_data) * 0.95)  ## For validation data (not train and test)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_data) - num_train])
    
    BATCH_SIZE = 64
    train_dataloader = DataLoader(split_train_, batch_size=len(split_train_), shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=len(split_valid_), shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, collate_fn=collate_batch)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
    model = TextClassificationModel(vocab_size=len(vocabObject), embed_dim=args.dims, num_class=len(labelEncoder.classes_)).to(device)
    
    train_df = pd.DataFrame()
    for idx, (text, label, offsets) in enumerate(train_dataloader):
        vectorEmbeddded = model.embedding(text, offsets)
        vectorEmbeddded = vectorEmbeddded.cpu().detach().numpy()
        row_to_append = pd.DataFrame(data=vectorEmbeddded)
        train_df = pd.concat([train_df, row_to_append])
        train_df['class'] = label.cpu().detach().numpy()
    train_df['type'] = ['train'] * len(train_df)
    print(train_df.head(5))
        
    test_df = pd.DataFrame()
    for idx, (text, label, offsets) in enumerate(test_dataloader):
        vectorEmbeddded = model.embedding(text, offsets)
        vectorEmbeddded = vectorEmbeddded.cpu().detach().numpy()
        row_to_append = pd.DataFrame(data=vectorEmbeddded)
        test_df = pd.concat([test_df, row_to_append])
        test_df['class'] = label.cpu().detach().numpy()
    test_df['type'] = ['test'] * len(test_df)
    print(test_df.head(5))

    valid_df = pd.DataFrame()
    for idx, (text, label, offsets) in enumerate(valid_dataloader):
        vectorEmbeddded = model.embedding(text, offsets)
        vectorEmbeddded = list(vectorEmbeddded.cpu().detach().numpy())
        row_to_append = pd.DataFrame(data=vectorEmbeddded)
        valid_df = pd.concat([valid_df, row_to_append])
        valid_df['class'] = label.cpu().detach().numpy()
    valid_df['type'] = ['valid'] * len(valid_df)
    print(valid_df.head(5))
    
    df = pd.DataFrame()
    df = pd.concat([df, train_df])
    df = pd.concat([df, test_df])
    df = pd.concat([df, valid_df])
    print(df.head(5))
    
    
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    df.to_csv(path_or_buf=args.outputfile, index=False)

    print("Done!")
