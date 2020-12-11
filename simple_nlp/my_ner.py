import torch
from transformers import *
# from keras.preprocessing.sequence import pad_sequences
import numpy as np
import threading
import time
import random
from simpletransformers.ner.ner_utils import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from torch_chatbot import MyDataLoader, tokenizer
from tqdm.auto import tqdm, trange
import logging
# if __name__ == "__main__":
#     first_tensor = torch.tensor([1,2,3])
#     second_tensor = torch.tensor([4, 5, 6])
#     ret = torch.vstack([first_tensor, second_tensor])
#     print(ret)
logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
device = 'cpu'
# checkpoint = torch.load('albert-tiny-chinese-3.pth')

pretrained = 'voidful/albert_chinese_tiny'


# train_data = [
#     [0, 'Simple', 'B-MISC'], [0, 'Transformers', 'I-MISC'], [0, 'started', 'O'], [1, 'with', 'O'], [0, 'text', 'O'], [0, 'classification', 'B-MISC'],
#     [1, 'Simple', 'B-MISC'], [1, 'Transformers', 'I-MISC'], [1, 'can', 'O'], [1, 'now', 'O'], [1, 'perform', 'O'], [1, 'NER', 'B-MISC']
# ]
entity, ignore = 'school', 'O'

entity_str = ['十中', '一中']
test_entity_str = ['三中']
ignore_str = '在哪里'




def build_dataset(entity_str, label_list, is_predict=False):
    if not is_predict:
        train_data = []
        for index, e in enumerate(entity_str):

            for s in e:
                train_data.append([index, s, entity])
            for i in ignore_str:
                train_data.append([index, i, ignore])
        train_df = pd.DataFrame(train_data, columns=[
                                'sentence_id', 'words', 'labels'])
        print(train_df)
        data = [
            InputExample(guid=sentence_id, words=sentence_df["words"].tolist(
            ), labels=sentence_df["labels"].tolist(),)
            for sentence_id, sentence_df in train_df.groupby(["sentence_id"])
        ]
         
    else:
        data = [
            InputExample(i, sentence, [label_list[0] for word in sentence])
            for i, sentence in enumerate(entity_str)
        ]
        for i, sentence in enumerate(entity_str):
            print(sentence, sentence.split())
    features = convert_examples_to_features(
        data,
        label_list,
        128,
        tokenizer,
        # XLNet has a CLS token at the end
        cls_token_at_end=False,  # bool(args.model_type in ["xlnet"]),
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,  # 2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        # RoBERTa uses an extra separator b/w pairs of sentences,
        # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        sep_token_extra=False,  # bool(args.model_type in ["roberta"]),
        # PAD on the left for XLNet
        pad_on_left=False,  # bool(args.model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,  # 4 if args.model_type in ["xlnet"] else 0,
        pad_token_label_id=pad_token_label_id,
        process_count=4,
        silent=False,  # args.silent,
        use_multiprocessing=True,  # args.use_multiprocessing,
        chunksize=500,  # args.multiprocessing_chunksize,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    train_dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return train_dataset




print(data)
tokenizer = BertTokenizer.from_pretrained(pretrained)
pad_token_label_id = CrossEntropyLoss().ignore_index

print(pad_token_label_id)
label_list = [entity, ignore]

train_dataset = build_dataset(entity_str, label_list)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=8,
    num_workers=4,
)

bert = AlbertForTokenClassification.from_pretrained(pretrained, num_labels=2, output_attentions=False,
                                                    output_hidden_states=True)
bert.to(device)
bert.zero_grad()
num_train_epochs = 2
train_iterator = trange(int(num_train_epochs),
                        desc="Epoch", disable=False, mininterval=0)

logger.info('begin to train!')
fp16 = False
for epoch_number, _ in enumerate(train_iterator):
    bert.train()
    train_iterator.set_description(
        f"Epoch {epoch_number + 1} of {num_train_epochs}\n")
    batch_iterator = tqdm(
        train_dataloader,
        desc=f"Running Epoch {epoch_number} of {num_train_epochs}\n",
        disable=False,
        mininterval=0,
    )
    for step, batch in enumerate(batch_iterator):
        print('batch-->', step)
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3],
        }

        if fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()
            with amp.autocast():
                outputs = bert(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]
        else:
            outputs = bert(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]

        current_loss = loss.item()
        batch_iterator.set_description(
            f"Epochs {epoch_number}/{num_train_epochs}. Running Loss: {current_loss:9.4f}"
        )

        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

to_predict="十中在哪"
test_dataset = build_dataset(to_predict, label_list, is_predict=True)

test_sampler = SequentialSampler(test_dataset)
eval_dataloader = DataLoader(
    test_dataset,
    sampler=test_sampler,
    batch_size=8,
    num_workers=4,
)
bert.eval()
out_label_ids = None
preds = None
for batch in tqdm(eval_dataloader, disable=False, desc="Running Prediction"):
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3],
        }

        if fp16:
            with amp.autocast():
                outputs = bert(**inputs)
                tmp_eval_loss, logits = outputs[:2]
        else:
            outputs = bert(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            out_input_ids = inputs["input_ids"].detach().cpu().numpy()
            out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            out_input_ids = np.append(out_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
            out_attention_mask = np.append(
                out_attention_mask, inputs["attention_mask"].detach().cpu().numpy(), axis=0,
            )
token_logits = preds
preds = np.argmax(preds, axis=2)

label_map = {i: label for i, label in enumerate(label_list)}
print('output', label_list, label_map, preds)
out_label_list = [[] for _ in range(out_label_ids.shape[0])]
preds_list = [[] for _ in range(out_label_ids.shape[0])]

for i in range(out_label_ids.shape[0]):
    for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
            out_label_list[i].append(label_map[out_label_ids[i][j]])
            preds_list[i].append(label_map[preds[i][j]])
split_on_space = False
if split_on_space:
    preds = [
        [{word: preds_list[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
        for i, sentence in enumerate(to_predict)
    ]
else:
    preds = [
        [{word: preds_list[i][j]} for j, word in enumerate(sentence[: len(preds_list[i])])]
        for i, sentence in enumerate(to_predict)
    ]

print(preds)