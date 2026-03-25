import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x 形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    
class Trans_Tagger(nn.Module):
    def __init__(self,vocab_size,tag_size,emb_dim=256, nhead=8, num_layers=3, hidden_dim=512,dropout=0.1):
        super(Trans_Tagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=nhead, 
            dropout=dropout,
            dim_feedforward=hidden_dim,
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, tag_size)

    def forward(self, x, mask=None):
            
            x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
            x = self.pos_encoder(x)
            
            output = self.transformer_encoder(x, src_key_padding_mask=mask)
            
            logits = self.fc(output)
            return logits