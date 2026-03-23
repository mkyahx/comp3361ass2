import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from collections import Counter

class NERDataset(Dataset):
    def __init__(self,label_source,data_source,mode,word2idx=None):
        self.mode = mode
        #load labels and sentences with tags
        with open(label_source, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        self.t2i = label_data
        self.sentences = []
        with open(data_source,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line) 
                    self.sentences.append(item)
        #padding and make embeddings
        if word2idx == None:
            word_count = Counter()
            for sentence in self.sentences:
                for token in sentence["tokens"]:
                    word_count[token] += 1
            self.word2idx = word_count.most_common()
        else:
            self.word2idx = word2idx
        self.token2idx = {"<pad>":0, "<UNK>":1}
        index = 2
        for word,count in self.word2idx:
            self.token2idx[word] = index
            index+=1
        self.idx2token = {value:key for key,value in self.token2idx.items()}
        return

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self,idx):
        item = self.sentences[idx]
        tokens = item['tokens']
        idxs = [self.token2idx.get(token, 1) for token in tokens]
        tags = item['tags']
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(tags, dtype=torch.long)

    def __check__(self):
        print(f"the NERDataset is in mode: {self.mode}")
        print(f'the tags are {self.t2i}')
        print(f"the sentences are {self.sentences}")

def collate(batch):
    tokens,tags = zip(*batch)
    tokens_padded = pad_sequence(tokens,batch_first = True, padding_value=0)
    tags_padded = pad_sequence(tags,batch_first = True, padding_value= -100)
    return tokens_padded, tags_padded



#---------------------------testing--------------------------#

'''
label_url = "./ontonotes5/dataset/label.json"
sentences_url = "./ontonotes5/dataset/train00.json"
my_dataset = NERDataset(label_url,sentences_url,"lstm")

train_loader = DataLoader(
    my_dataset,
    batch_size = 4,
    shuffle = True,
    collate_fn = collate
)

for batch_tokens, batch_tags in train_loader:
    print(f"Batch Tokens Shape: {batch_tokens.shape}") # 应该是 [4, 当前 batch 最长长度]
    print(f"Batch Tags Shape: {batch_tags.shape}")
    print("First sentence padded indices:", batch_tokens[0])
    break

'''