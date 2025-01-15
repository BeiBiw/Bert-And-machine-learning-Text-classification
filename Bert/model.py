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
        # bert的输出结果有四个维度：
        # last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态。
        # 此时shape是(batch_size, sequence_length, hidden_size)，[:,0,:]的意思是取出第一个也就是cls对应的结果
        outputs = hiden_outputs[0][:, 0, :]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
        output = self.fc(outputs)
        return output
