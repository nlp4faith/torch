import torch
from transformers import *
from keras.preprocessing.sequence import pad_sequences

# from torch_chatbot import MyDataLoader, tokenizer

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

checkpoint = torch.load('albert-tiny-chinese-3.pth')

pretrained = 'voidful/albert_chinese_tiny'
bert = AlbertForSequenceClassification.from_pretrained(pretrained, num_labels=2, output_attentions=False,
                                                       output_hidden_states=False)


# print(checkpoint)
bert.load_state_dict(checkpoint['model'])

bert.to(device)

tokenizer = BertTokenizer.from_pretrained(pretrained)
# inputtext = "1989年64"


def evaluate(inputtext):
    input_ids = tokenizer.encode(inputtext, add_special_tokens=True)
    input_ids = pad_sequences([input_ids], maxlen=128, dtype="long",
                              value=0, truncating="post", padding="post")
    input_ids = input_ids.ravel()
    att_mask = [int(token_id > 0) for token_id in input_ids]
    print(input_ids, att_mask)
    with torch.no_grad():
        mask = torch.tensor(att_mask).to(device)
        mask = mask.unsqueeze(0)
        input = torch.tensor(input_ids).to(device)
        input = input.unsqueeze(0)
        outputs = bert(input,
                       token_type_ids=None,
                       attention_mask=mask)
        print(outputs)


# data_loader = MyDataLoader(
#     ['text/test_weibo_data.txt', 'text/test_politice_comments.txt'])

# data_loader.load_data()
# data_loader.preprocess(tokenizer)
# # data_loader.load_npy()
# data_loader.generate_dataset()
# print(data_loader)

while(1):
    try:
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit':
            break
        # Normalize sentence
        # input_sentence = normalizeString(input_sentence)
        # Evaluate sentence
        output_words = evaluate(input_sentence)
        # Format and print response sentence
        # output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        # print('Bot:', ' '.join(output_words))

    except KeyError:
        print("Error: Encountered unknown word.")
