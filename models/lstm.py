import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

class BiLSTM_Tagger(nn.Module):
    def __init__(self,vocab_size, tag_size,emb_dim,hidden_dim):
        super(BiLSTM_Tagger,self).__init__()
        self.embedding = nn.Embedding(vocab_size,emb_dim)
        self.model = nn.LSTM(emb_dim, hidden_dim // 2, 
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim,tag_size)
    def forward(self,x):
        emb = self.embedding(x)
        y,_ = self.model(emb)
        logit = self.hidden2tag(y)
        return logit
