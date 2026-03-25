import transformers
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch.nn as nn


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding='max_length', max_length=128)
    
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
class DistilBert_Tagger(nn.Module):
    def __init__(self, tag_size, dropout=0.1):
        super(DistilBert_Tagger, self).__init__() 
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.dropout = nn.Dropout(dropout)
        #Added Linear layer
        self.fc = nn.Linear(self.distilbert.config.hidden_size, tag_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.fc(sequence_output)
        return logits