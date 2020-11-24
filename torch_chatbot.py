from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
import time
from torch.nn.functional import softmax
from transformers import *


import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

print(USE_CUDA, device)


pretrained = 'voidful/albert_chinese_tiny'


class MyDataLoader(object):

    __DEFAULT_MAX_LEN = 128

    def __init__(self, files):
        self.files = files
        self.datum, self.labels = None, None

    def load_data(self):
        for i, file_name in tqdm(enumerate(self.files)):
            dataset = pd.read_csv(file_name, sep=',')
            label, data = dataset.iloc[:, 0].to_numpy(
            ).reshape(-1, 1), dataset.iloc[:, 1:].to_numpy()
            print(label.shape, data.shape)
            if self.datum is None:
                self.datum = data
                self.labels = label
            else:
                self.datum = np.concatenate([self.datum, data], axis=0)
                self.labels = np.concatenate([self.labels, label], axis=0)
            # self.labels.append(label)
        # self.datum = np.array(datum).reshape(-1, 1)
        # self.labels = np.array(labels).reshape(-1, 1)
        self.num_labels = len(np.unique(self.labels))
        # self.labels = self.labels.ravel()
        print(self.datum.shape, self.labels.shape, self.num_labels)
        clean_datum, clean_label= [], []
        for i, sen in enumerate(self.datum):
            if type(sen[0]) == str:
                clean_datum.append(sen[0])
                clean_label.append(self.labels[i])
        self.datum = clean_datum
        self.labels = clean_label
        str_len = np.array([len(sen) for sen in self.datum])
        self.MAX_LEN = min(np.max(str_len), self.__DEFAULT_MAX_LEN)
        print('datum-->', self.MAX_LEN)

        # self.datum = self.datum[:100]
        # self.labels = self.labels[:100]
        

    def preprocess(self, tokenizer, is_save=True, debug_split=2):
        self.input_ids, self.tokenizer = tokenizer(self.datum, self.MAX_LEN)
        
        print('\nPadding/truncating all sentences to %d values...' % self.MAX_LEN)

        print('\nPadding token: "{:}", ID: {:}'.format(
            self.tokenizer.pad_token, self.tokenizer.pad_token_id))
        # print(self.input_ids)
        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        self.input_ids = pad_sequences(self.input_ids, maxlen=self.MAX_LEN, dtype="long",
                                       value=0, truncating="post", padding="post")
        # print(self.input_ids)
        self.attention_masks = []

        # For each sentence...
        for sent in self.input_ids:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            self.attention_masks.append(att_mask)
        # print('-->', self.input_ids.shape,self.attention_masks.shape, self.labels.shape) 
        if is_save:
            np.save('input_ids.npy', self.input_ids)
            np.save('attention_masks.npy', self.attention_masks)
            np.save('labels.npy', self.labels)
        print('--')

    def load_npy(self):
        self.input_ids = np.load('input_ids.npy', allow_pickle=True)
        self.attention_masks = np.load('attention_masks.npy', allow_pickle=True)
        self.labels = np.load('labels.npy', allow_pickle=True)
        self.num_labels = len(np.unique(self.labels))


    def generate_dataset(self, batch_size=32):

        self.batch_size = batch_size

        train_inputs = torch.tensor(self.input_ids)
        train_masks = torch.tensor(self.attention_masks)
        train_labels = torch.tensor(self.labels)

        print('-->train', train_inputs.size(), train_masks.size(), train_labels.size())
        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size, num_workers=6)

        print(len(self.train_dataloader))

        # Create the DataLoader for our validation set.
        # validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        # validation_sampler = SequentialSampler(validation_data)
        # validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


data_loader = MyDataLoader(
    ['text/train_weibo_data.txt', 'text/train_politice_comments.txt'])



def tokenizer(data, max_len):
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    input_ids = []
    for sentence in data:
        if type(sentence) == str:
            input_ids.append(tokenizer.encode(
                sentence[:max_len], add_special_tokens=True))
        else:
            print(sentence)
    return input_ids, tokenizer

# data_loader.load_data()
# data_loader.preprocess(tokenizer)
data_loader.load_npy()
data_loader.generate_dataset()


bert = AlbertForSequenceClassification.from_pretrained(pretrained, num_labels=data_loader.num_labels, output_attentions=False,  # Whether the model returns attentions weights.
                                                       output_hidden_states=False)
bert.cuda()


# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(bert.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )
# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(data_loader.train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train():
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        bert.train()

        # For each batch of training data...
        for step, batch in enumerate(data_loader.train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(data_loader.train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            bert.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = bert(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(data_loader.train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(
            format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        bert.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        # for batch in validation_dataloader:

        # # Add batch to GPU
        # batch = tuple(t.to(device) for t in batch)

        # # Unpack the inputs from our dataloader
        # b_input_ids, b_input_mask, b_labels = batch

        # # Telling the model not to compute or store gradients, saving memory and
        # # speeding up validation
        # with torch.no_grad():

        #     # Forward pass, calculate logit predictions.
        #     # This will return the logits rather than the loss because we have
        #     # not provided labels.
        #     # token_type_ids is the same as the "segment ids", which
        #     # differentiates sentence 1 and 2 in 2-sentence tasks.
        #     # The documentation for this `model` function is here:
        #     # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        #     outputs = bert(b_input_ids,
        #                     token_type_ids=None,
        #                     attention_mask=b_input_mask)

        # # Get the "logits" output by the model. The "logits" are the output
        # # values prior to applying an activation function like the softmax.
        # logits = outputs[0]

        # # Move logits and labels to CPU
        # logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

        # # Calculate the accuracy for this batch of test sentences.
        # tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # # Accumulate the total accuracy.
        # eval_accuracy += tmp_eval_accuracy

        # # Track the number of batches
        # nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        # print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

        all_data = dict(
        optimizer=optimizer.state_dict(),
        model=bert.state_dict(),
        info=u'albert-tiny-chinese'
        )
        torch.save(all_data, '{}-{}.pth'.format(all_data['info'], epoch_i))


        # torch.save({
        #         'iteration': iteration,
        #         'en': encoder.state_dict(),
        #         'de': decoder.state_dict(),
        #         'en_opt': encoder_optimizer.state_dict(),
        #         'de_opt': decoder_optimizer.state_dict(),
        #         'loss': loss,
        #         'voc_dict': voc.__dict__,
        #         'embedding': embedding.state_dict()
        #     }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


    print("")
    print("Training complete!")




if __name__ == "__main__":
    train()

# class Model(nn.Module):

#     def __init__(self, config):
#         super(Model, self).__init__()

#         self.tokenizer = BertTokenizer.from_pretrained(pretrained)
#         self.bert = AlbertModel.from_pretrained(pretrained)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         self.fc = nn.Linear(config.hidden_size, config.num_classes)

#     def forward(self, x):
#         context = x[0]  # 输入的句子
#         mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
#         _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
#         out = self.fc(pooled)
#         return out

# def model():
#     bert = AlbertForSequenceClassification.from_pretrained(pretrained, num_labels=2, output_attentions=False,  # Whether the model returns attentions weights.
#                                                         output_hidden_states=False)
#     bert.cuda()
#     tokenizer = BertTokenizer.from_pretrained(pretrained)
#     print(tokenizer.encode("我是谁", add_special_tokens=False))
#     params = list(bert.named_parameters())
#     for p in params:
#         print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


#     optimizer = AdamW(bert.parameters(),
#                     lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                     eps=1e-8  # args.adam_epsilon  - default is 1e-8.
#                     )

#     # Number of training epochs (authors recommend between 2 and 4)
#     epochs = 4

#     # Total number of training steps is number of batches * number of epochs.
#     total_steps = len(train_dataloader) * epochs

#     # Create the learning rate scheduler.
#     scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                 num_warmup_steps=0,  # Default value in run_glue.py
#                                                 num_training_steps=total_steps)


# AlbertForMaskedLM
# inputtext = "今天[MASK]情很好"

# # maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

# input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).to(device)

# input_ids = input_ids.unsqueeze(0)  # Batch size 1


# with torch.no_grad():
#     outputs = model(input_ids)
#     # print(outputs)

# print(input_ids.size(), model(input_ids)[0])

# outputs = model(input_ids, masked_lm_labels=input_ids)
# loss, prediction_scores = outputs[:2]
# logit_prob = softmax(prediction_scores[0, maskpos]).data.tolist()
# predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token,logit_prob[predicted_index])

# from transformers import pipeline
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# ret = classifier('虽然我们输了，但是我们很开心')
# print(ret)
