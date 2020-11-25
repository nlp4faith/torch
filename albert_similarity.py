import torch
from transformers import *
from keras.preprocessing.sequence import pad_sequences
import numpy as np
# from torch_chatbot import MyDataLoader, tokenizer

# if __name__ == "__main__":
#     first_tensor = torch.tensor([1,2,3])
#     second_tensor = torch.tensor([4, 5, 6])
#     ret = torch.vstack([first_tensor, second_tensor]) 
#     print(ret)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

checkpoint = torch.load('albert-tiny-chinese-3.pth')

pretrained = 'voidful/albert_chinese_tiny'
bert = AlbertModel.from_pretrained(pretrained, num_labels=2, output_attentions=False,
        output_hidden_states=True)
bert.to(device)
bert.eval()

tokenizer = BertTokenizer.from_pretrained(pretrained)

# params = list(bert.named_parameters())
# for p in params:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
# masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
# masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
# tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

def evaluate(inputtext):
    encoding = tokenizer(inputtext, return_tensors='pt', padding=True, return_length=True)

    # print(input_ids, att_mask)
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        # input_len = encoding['length'].to(self.device)
        mask = encoding['attention_mask'].to(device)
        outputs = bert(input_ids, attention_mask=mask)
        ret = outputs[0]
        
        cpu_ret = ret.cpu().detach().numpy()
        cpu_mask = mask.cpu().detach().numpy()
        ret = np.dot(cpu_mask, cpu_ret.squeeze(0))
        # ret = torch.sum(ret, 1)
        return torch.tensor(ret)


def evaluate_bak(inputtext):
    input_ids = tokenizer.encode(inputtext, add_special_tokens=True)
    input_ids = pad_sequences([input_ids], maxlen=128, dtype="long",
                              value=0, truncating="post", padding="post")
    input_ids = input_ids.ravel()
    att_mask = [int(token_id > 0) for token_id in input_ids]
    # print(input_ids, att_mask)
    with torch.no_grad():
        mask = torch.tensor(att_mask).to(device)
        mask = mask.unsqueeze(0)

        input = torch.tensor(input_ids).to(device)
        input = input.unsqueeze(0)
        outputs = bert(input, attention_mask=mask)
        ret = outputs[0]
        
        cpu_ret = ret.cpu().detach().numpy()
        cpu_mask = mask.cpu().detach().numpy()
        ret = np.dot(cpu_mask, cpu_ret.squeeze(0))
        # ret = torch.sum(ret, 1)
        return torch.tensor(ret)

def evaluate_old(inputtext):
    input_ids = tokenizer.encode(inputtext, add_special_tokens=True)
    input_ids = pad_sequences([input_ids], maxlen=128, dtype="long",
                              value=0, truncating="post", padding="post")
    input_ids = input_ids.ravel()
    att_mask = [int(token_id > 0) for token_id in input_ids]
    # print(input_ids, att_mask)
    with torch.no_grad():
        mask = torch.tensor(att_mask).to(device)
        mask = mask.unsqueeze(0)
        input = torch.tensor(input_ids).to(device)
        input = input.unsqueeze(0)
        outputs = bert(input,
                       token_type_ids=None,
                       attention_mask=mask)
        # print(outputs)
        attention_mask = mask.eq(0)
        result = outputs.last_hidden_state.masked_fill_(attention_mask.unsqueeze(-1), float(0))
        return torch.div(torch.sum(result, dim=1), 128)  
        # return outputs

def test1():
    # s1 = evaluate('今天吃饭了吗')
    # question_ts = evaluate('时代香海彼岸的学区房')
    # templates = ['请问一下二中的学区房是什么', '请问一下一中的学区房是什么', '请问一下斗门中学的学区房是什么', '斗门时代香海的学区房', '斗中学区房', '时代彼岸的学区房', '时代比岸的学区房']
    # templates_ts = []
    # for t in templates:
    #     templates_ts.append(evaluate(t))
 
    # s2 = torch.vstack(templates_ts)
    # ret = torch.nn.functional.cosine_similarity(question_ts, s2, dim=1, eps=1e-8)    
    # print(ret)
    question = "玲珑密保锁如何冻结账号"
    question_ts = evaluate(question)
    templates = ["玲珑锁冻结账号解绑了密保锁","玲珑锁如何冻结账号","玲珑锁账号该怎么冻结"]
    for template in templates:
        ret = torch.nn.functional.cosine_similarity(question_ts, evaluate(template), dim=1, eps=1e-8)    
        print(template, ret)

if __name__ == "__main__":
    test1()
