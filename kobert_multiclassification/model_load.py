#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import transformers

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset 
import gluonnlp as nlp
import numpy as np
from collections import defaultdict

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import BertTokenizer, BertModel

from pathlib import Path
from utils import BERTClassifier

import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    


# In[ ]:


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 7,
                 dr_rate = None,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids = torch.zeros_like(segment_ids).long(), attention_mask=attention_mask.float().to(token_ids.device))

        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler

        return self.classifier(out)


# In[ ]:


import re

# 측정할 사용자 이름 적으면 된다.(사용자로부터 받아와야한다.)
user_name="신상윤"

#대화 리스트 들어갈 곳
all_conversation_arr=[]

# 요일 없애야하는것
remove_characters="년월일-월화수목금토요 " 
cur_time=""
conversation=""

#여기다가 원하는 카카오 데이터 셋(txt) 집어 넣으면 됨 ( 여기서는 KakaoTalk_20210426_2042_40_926_김준홍.txt 를 넣었음)
f = open("KakaoTalk_20210503_1909_44_475_신상윤.txt", 'r',encoding='UTF8')
while True:
    line = f.readline()
    if not line:  #마지막에 도달했을 때 반복문 빠져나옴
        break
    #요일이 시작되는 경우
    if line[:5]=="-----": 
        line=''.join(x for x in line if x not in remove_characters)
        cur_time=line #  년, 월, 일 형태로 받는다.(ex 2020120)
        while True:
            line = f.readline()
            # line 이 빈 값일 때 or 끝났을 때
            if line[:5]=="-----" or not line:
                break
            #사용자의 이름인 것만 받아온다. (한줄이 50정도max -> 한줄만 받자)
            if line[1:len(user_name)+1]==user_name:
                conversation=line[16:]
                all_conversation_arr.append([user_name,cur_time,conversation])
                

for i in range(len(all_conversation_arr)):
    all_conversation_arr[i][1]=re.sub("\n","",all_conversation_arr[i][1])
    all_conversation_arr[i][2]=re.sub("\n","",all_conversation_arr[i][2])

# 이상한 문자 있는 문장  지워버림 - 훨씬 깔끔하게 나옴
remove_letters="0123456789ㅂㅈㄷㄱㅅㅕㅑㅐㅔ[ㅁㄴㅇㅃㅉㄸㄲㅆㄹㅎ,_ㅗㅓㅏ※ㅣ;]'ㅋㅌㅊ)=(ㅠㅜㅍㅡabcdefghijklmnopqrstuvwxyz/QWERTYUIOPASDFGHJKLZXCVBNM#%-\":"
for i in reversed(range(len(all_conversation_arr))):
    for x in all_conversation_arr[i][2]:
        if x in remove_letters:
            del all_conversation_arr[i]
            break

# 길이가 2 이하인 문자열 제거
for i in reversed(range(len(all_conversation_arr))):
    if len(all_conversation_arr[i][2])<=10 or len(all_conversation_arr[i][2]) > 56:
        del all_conversation_arr[i]
        
all_conversation_arr.reverse()
# 최근 대화 50개 리스트
num_50_conversation=all_conversation_arr[:50]
# 최근 대화 100개 리스트
num_100_conversation=all_conversation_arr[:100]
    
f.close()


# In[ ]:


def main():
    # _model_dir = "/home/k4ke/kobert/saves"
    # model_dir = Path(_model_dir)
    # model_config = Config(json_path = model_dir / 'config.json')

    # Vocab and Tokenizer
    tokenizer = get_tokenizer()
    
    bertmodel, vocab = get_pytorch_kobert_model()
    # token_to_idx = vocab.token_to_idx
    #
    # # vocab_size = len(token_to_idx)
    # print("len(toekn_to_idx): ", len(token_to_idx))
    #
    # with open(model_dir / "token2idx_vocab.json", 'w', encoding='utf-8') as f:
    #     json.dump(token_to_idx, f, ensure_ascii=False, indent=4)
    #
    # # save vocab & tokenizer
    # with open(model_dir / "vocab.pkl", 'wb') as f:
    #     pickle.dump(vocab, f)
    #
    # # load vocab & tokenizer
    # with open(model_dir / "vocab.pkl", 'rb') as f:
    #     vocab = pickle.load(f)

    # tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=64)
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    model = BERTClassifier(bertmodel)

    # load model
    model_dict = model.state_dict()
    # checkpoint = torch.load("./experiments/base_model_with_crf_val/best-epoch-12-step-1000-acc-0.960.bin", map_location=torch.device('cpu'))
    # checkpoint = torch.load("/home/k4ke/kobert/saves/best-epoch-5-f1-0.916.bin", map_location = torch.device('cpu'))
    checkpoint = torch.load("C:/Users\KIMJOONHONG/ml/datasets/saves/best-epoch-36-f1-0.732.bin", map_location=torch.device('cpu'))
    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys)
    # model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    #model.resize_token_embeddings(len(tokenizer))
    emo_dict = {0: '공포', 1: '놀람', 2: '분노', 3: '슬픔', 4: '중립', 5: '행복', 6: '혐오'}
    emo_dict2 = {0: '공포', 1: '분노', 2: '슬픔', 3: '행복', 4: '혐오'}
    result=np.zeros((1,7),dtype=float)
    #print(arr)
    for i in all_conversation_arr:
    #while True:
        #_sentence = input("input: ")
        _sentence=str(i[2])
        if _sentence == '-1':
            break
        transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=64, pad=True, pair=False)
        # self.sentences = [transform([i[sent_idx]]) for i in dataset]
        sentence = [transform([_sentence])]
        #data_train = BERTDataset(sentence, 0, 1, tok, 64, True, False)
        dataloader = torch.utils.data.DataLoader(sentence, batch_size=1)
        _token_ids = dataloader._index_sampler.sampler.data_source
        # print(_token_ids)
        # print(_token_ids[0])
        # print(_token_ids[0][0])
        _t = torch.from_numpy(_token_ids[0][0])
        _t = _t.tolist()
        token_ids = torch.tensor(_t, dtype=torch.long).unsqueeze(0).cuda()
        val_len = torch.tensor([len(token_ids[0])], dtype=torch.long).cuda()
        # val_len = torch.tensor([len(token_ids)], dtype=torch.long).cuda()

        _s = torch.from_numpy(_token_ids[0][1])
        _s = _s.tolist()
        segment_ids = torch.tensor(_s, dtype=torch.long).unsqueeze(0).cuda()
        # segment_ids = torch.from_numpy(_token_ids[0][1]).unsqueeze(0)
        # segment_ids = torch.zeros()
        # print(len(token_ids)) # 1

        out = model(token_ids, val_len, segment_ids)
        out_idx = np.argmax(out.cpu().detach().numpy())
        softmax = nn.Softmax(dim=1)
        score = softmax(out).cpu().detach().numpy()
        result+=score
        #print("out: ", out)
        print("input:",_sentence)
        print(out_idx, emo_dict[out_idx])
        #print("score: ", score)
    print(emo_dict2[np.delete(result,[1,4]).argmax()])
    
if __name__ == "__main__":
    main()
    print("done")

