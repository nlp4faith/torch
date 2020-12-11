import torch
from transformers import *
# from keras.preprocessing.sequence import pad_sequences
import numpy as np
import threading, time, random
from torch.autograd import Variable
import os
from transformers.convert_pytorch_checkpoint_to_tf2 import *
# from torch_chatbot import MyDataLoader, tokenizer
import tensorflow as tf
# if __name__ == "__main__":
#     first_tensor = torch.tensor([1,2,3])
#     second_tensor = torch.tensor([4, 5, 6])
#     ret = torch.vstack([first_tensor, second_tensor]) 
#     print(ret)
from bert4keras.models import build_transformer_model

def test1():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    device = 'cpu'
    # checkpoint = torch.load('/home/faith/torch_tutorials/albert_chinese_tiny/pytorch_model.bin')
    # print(checkpoint)
    # checkpoint = torch.load('albert-tiny-chinese-3.pth')

    pretrained = 'voidful/albert_chinese_tiny'
    pretrained = '/home/faith/torch_tutorials/albert_chinese_tiny'
    pretrained = '/home/faith/ALBERT/pytorch_model.bin'
    # bert = AlbertModel.from_pretrained(pretrained, num_labels=2, output_attentions=False,
            # output_hidden_states=True)
    model_type = 'albert'
    # convert_pt_checkpoint_to_tf(model_type=model_type, pytorch_checkpoint_path=pretrained,config_file='/home/faith/ALBERT/config.json',tf_dump_path="albert-tf_model.h5", compare_with_pt_model=True)

    pt_state_dict = torch.load(pretrained)
    # old_keys = []
    # new_keys = []
    # for key in pt_state_dict.keys():
    #     new_key = None
    #     if "gamma" in key:
    #         new_key = key.replace("gamma", "weight")
    #     if "beta" in key:
    #         new_key = key.replace("beta", "bias")
    #     if new_key:
    #         old_keys.append(key)
    #         new_keys.append(new_key)
    # for old_key, new_key in zip(old_keys, new_keys):
    #     pt_state_dict[new_key] = pt_state_dict.pop(old_key)

    # for name in pt_state_dict:
    #     print(name)

    # model = RobertaModel

    config_file='/home/faith/ALBERT/config.json'
    # config_class = AlbertConfig
    config = AlbertConfig.from_json_file(config_file)
    tf_model = TFAlbertModel(config)

    token_ids = np.zeros(512, dtype=int) # np.array([ 101, 1506,  102])
    print(token_ids.shape)
    token_ids[0] = 101
    token_ids[1] = 1506
    token_ids[2] = 102


    masks = np.ones(512, dtype=int) #np.array([ 0, 0,  0])
    masks[:3] = 0

    masks = masks.reshape(1, -1)
    token_ids = token_ids.reshape(1, -1)
    print(masks.shape, token_ids.shape)



    # token_ids = token_ids.tolist()
    # masks = masks.tolist()
   
    # imported = tf.saved_model.load('alert_saved_model_new2')

    # params = {'inputs_ids': tf.Tensor(token_ids), 'attention_mask': tf.Tensor(masks)}
    # params = [tf.Tensor(token_ids), tf.Tensor(masks)]

    # out1 = imported.predict(tf.Tensor(token_ids))
    # print(imported.signatures["serving_default"].structured_outputs)


    # output = tf_model([token_ids, masks], training=False)
    # output = imported([token_ids, masks], training=False)
    # print('--')

    # print(output.last_hidden_state)
    # TFAlbertPreTrainedModel
    # tf_inputs = tf_model.dummy_inputs
    # print(tf_inputs)

    # tf_model(tf_inputs, training=False)  # Make sure model is built


    # load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path)
    # load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=True)

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        print(sw_name)



    tf_model = load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=[token_ids, masks])
    # output2 = tf_model([token_ids, masks], training=False)
    # output2 = tf_model(np.array([[101, 1506, 102]]), training=False)
    print('---')
    tf_model.summary()
    print('---')
    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        print(sw_name)

    # output3 = tf_model([np.array(token_ids),np.array(masks)])


    output2 = tf_model(np.array([[101, 1506, 102]]), training=False)
    # print('--', output2)
    # tf_model.summary()
    # # tf_model.save('alert_saved_model_new')
    # from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

    # print(tf_model.input)
    # signature = predict_signature_def(inputs={'input1': tf_model.input[0],
    #                                       'input2': tf_model.input[1]})

    tf.saved_model.save(tf_model, 'alert_saved_model_new999')

    # , signatures=tf_model.call.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_1"))
    print('----', 'begin saving', type(tf_model))
    # tf.keras.models.save_model(tf_model, 'alert_saved_model_new5')



def test2():
    #/home/faith/00000123
    imported = tf.saved_model.load('alert_saved_model_new3')
    imported = tf.keras.models.load_model('/home/faith/00000123')
    print(imported.signatures["serving_default"].structured_outputs)
    print(imported.signatures["serving_default"].structured_input_signature)
    print(imported.signatures["serving_default"].__dict__)

    # symbolic_weights = imported.trainable_weights + imported.non_trainable_weights
    # for symbolic_weight in symbolic_weights:
    #     sw_name = symbolic_weight.name
    #     print(sw_name)
    # imported.summary()
    imported([np.array([[101, 1506, 102, 0]], dtype=np.int32), np.array([[1, 1, 1, 0]], dtype=np.int32)])



def test3():
    # tf_model = tf.saved_model.load('alert_saved_model_new5') #alert_saved_model_new5
    # tf_model.summary()
    tf_model = tf.saved_model.load('new_alert_4_keras_2') #alert_saved_model_new5
    # tf_model = tf.keras.models.load_model('new_alert_4_keras_2')
    outputs = tf_model.signatures['serving_default']
    outputs.inputs[0].set_shape([None, None])
    print(outputs.structured_input_signature)
    print(outputs.inputs)

    # symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    # for symbolic_weight in symbolic_weights:
    #     sw_name = symbolic_weight.name
    #     print(sw_name)

    token_ids = np.zeros(3, dtype=np.float32) # np.array([ 101, 1506,  102])
    print(token_ids.shape)
    token_ids[0] = 101
    token_ids[1] = 1506
    token_ids[2] = 102


    masks = np.ones(3, dtype=np.float32) #np.array([ 0, 0,  0])
    masks[:3] = 0

    masks = masks.reshape(1, -1)
    token_ids = token_ids.reshape(1, -1)
    # tf_model.summary()
    # output2 = tf_model([np.array([101, 1506, 102]).reshape(1, -1), np.array([0, 0, 0]).reshape(1, -1)])
    # output2 = tf_model({'input_1':np.array(token_ids), 'input_2':np.array(masks)})

    output2 = tf_model([np.array(token_ids),np.array(masks)])
    # tf_model({'input_ids': np.array([[101, 1506, 102, 0, 0]])})
    # output2 = tf_model({'input_ids':np.array([[101, 1506, 102, 0]], dtype=np.int32)})
    # output2 = tf_model([np.array([[101, 1506, 102, 0]], dtype=np.int32), np.array([[1, 1, 1, 0]], dtype=np.int32)])

    # output2 = tf_model({'input_1':np.array(token_ids), 'input_2':np.array(masks)})

    # output2 = tf_model.predict([ np.array([101, 1506, 102]) ,  np.array([0, 0, 0]) ] )
    print('--', output2)

def test4():
    config_path = '/home/faith/albert_tiny_google_zh_489k/albert_config.json'
    checkpoint_path = '/home/faith/albert_tiny_google_zh_489k/albert_model.ckpt'
    dict_path = '/home/faith/albert_tiny_google_zh_489k/vocab.txt'
    
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='albert',
        return_keras_model=False,
    )
    tf_model = bert.model
    tf_model.summary()
    print(type(tf_model))
    # tf_model.save('new_alert_4_keras')
    tf.saved_model.save(tf_model, 'new_alert_4_keras_3')


def test5():
    import tensorflow as tf
    from tensorflow import keras
    def get_model():
    # Create a simple model.
        inputs = keras.Input(shape=(32,))
        outputs = keras.layers.Dense(1)(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model
    model = get_model()
    model.summary()
    print(type(model))
    model.save("my_model1")


def test_roberta():
    roberta_path = '/home/faith/roberta_tiny_pair'
    config_path = roberta_path + '/bert_config.json'
    checkpoint_path = roberta_path + '/bert_model.ckpt'
    tokenizer = BertTokenizer.from_pretrained(roberta_path, add_special_tokens=False)
    print(tokenizer)
    # configuration = RobertaConfig()
    # print(configuration)
    model = BertModel.from_pretrained(roberta_path,  from_tf=True)
    print(model)
    print(type(model))
    torch.save(model, 'roberta_tiny')
    
    # bert = build_transformer_model(
    #     config_path=config_path,
    #     checkpoint_path=checkpoint_path,
    #     model='roberta',
    #     return_keras_model=False,
    # )
    # tf_model = bert.model
    # tf_model.summary()


def test_albert():
    pretrained = '/home/faith/ALBERT'
    bert = AlbertModel.from_pretrained(pretrained, output_attentions=False, output_hidden_states=True)
    bert.eval()
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    print(bert)
    print(tokenizer)

def test_load_roberta():
    model = torch.load('roberta_tiny')
    print(model)
    states = model.state_dict()
    print(states)
    print('---')
    for s in states:
        print(s)

if __name__ == "__main__":
    # test2()
    # test1()
    # test3()
    # test4()
    # test5()
    # test_roberta()
    test_load_roberta()
    # test_albert()
    