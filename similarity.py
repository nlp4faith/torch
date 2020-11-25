from transformers import BertTokenizer, AutoTokenizer, AutoModel, BertModel, AlbertModel
import numpy as np
import torch
import torch.nn as nn
import time


class Similarity(object):
    def __init__(self):
        # model_name = "models/roberta-wwm-ext/"
        model_name = "models/ALBERT/"
        self.tokenizer = BertTokenizer.from_pretrained(model_name, add_special_tokens=False)
        # self.model = BertModel.from_pretrained(model_name, output_hidden_states=True, return_dict = True)
        self.model = AlbertModel.from_pretrained(model_name, output_hidden_states=True, return_dict = True)
        self.model.eval()
        self.device = "cuda:0"
        

    def sentence2vec(self, sentence: str):
        encoding = self.tokenizer(sentence, return_tensors='pt', padding=True, return_length=True)  # , add_special_tokens=False)
        input_ids = encoding['input_ids'].to(self.device)
        input_len = encoding['length'].to(self.device)
        input_len = input_len.type(torch.float64).unsqueeze(1)
        attention_mask = encoding['attention_mask'].to(self.device)
        model = self.model
        model.to(self.device)
        outputs = model(input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.eq(0)
        result = outputs.last_hidden_state.masked_fill_(attention_mask.unsqueeze(-1), float(0))
        return torch.div(torch.sum(result, dim=1), input_len).cpu()

    def get_similarity(self, question: str, q: list):
        s1 = self.sentence2vec(question)
        s2 = self.sentence2vec(q)
        result = torch.nn.functional.cosine_similarity(s1.repeat(s2.size(0), 1), s2, dim=1, eps=1e-8)
        return list(result.detach().numpy())

if __name__ == "__main__":
    question = "玲珑密保锁如何冻结账号"
    q = ["玲珑锁冻结账号解绑了密保锁","玲珑锁如何冻结账号","玲珑锁账号该怎么冻结"]
    similarity = Similarity()
    print(similarity.get_similarity(question,q))
