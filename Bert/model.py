import torch.nn as nn
import torch
from transformers import BertModel,BertTokenizer

class BertClassfication(nn.Module):
    def __init__(self,config):
        super(BertClassfication, self).__init__()
        self.model_name = 'F:/bert-base-uncased/'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768, config.num_classes)

    def forward(self, input_ids,attention_mask):
        hiden_outputs = self.model(input_ids, attention_mask=attention_mask)
        # last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768
        # shape(batch_size, sequence_length, hidden_size)
        outputs = hiden_outputs[0][:, 0, :]
        output = self.fc(outputs)
        return output
