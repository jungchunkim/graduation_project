
from torch.utils.data import Dataset
import re
import pandas as pd
from transformers import BertTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
import numpy as np
import gluonnlp as nlp
from torch import nn
import torch
import transformers
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
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

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=torch.zeros_like(
            segment_ids).long(), attention_mask=attention_mask.float().to(token_ids.device))

        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler

        return self.classifier(out)


# In[5]:


class KoBERTPredictor:
    def __init__(self, model_path="C:/Users\KIMJOONHONG/ml/datasets/best-epoch-36-f1-0.732.bin"):

        tokenizer = get_tokenizer()
        bertmodel, vocab = get_pytorch_kobert_model()
        self.tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        self.model = BERTClassifier(bertmodel)

        # load model
        model_dict = self.model.state_dict()
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        convert_keys = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key_name = k.replace("module.", '')
            if new_key_name not in model_dict:
                print("{} is not int model_dict".format(new_key_name))
                continue
            convert_keys[new_key_name] = v

        self.model.load_state_dict(convert_keys)

        self.model.eval()
        self.model.to(device)

    def predict(self, user_name, conversation_path):
        # ?????? ????????? ????????? ???
        all_conversation_arr = []

        # ?????? ??????????????????
        remove_characters = "?????????-????????????????????? "
        cur_time = ""
        conversation = ""
        # ???????????? ????????? ????????? ????????? ???(txt) ?????? ????????? ??? ( ???????????? KakaoTalk_20210426_2042_40_926_?????????.txt ??? ?????????)
        f = open(conversation_path, 'r', encoding='UTF8')
        while True:
            line = f.readline()
            if not line:  # ???????????? ???????????? ??? ????????? ????????????
                break
            # ????????? ???????????? ??????
            if line[:5] == "-----":
                line = ''.join(x for x in line if x not in remove_characters)
                cur_time = line  # ???, ???, ??? ????????? ?????????.(ex 2020120)
                while True:
                    line = f.readline()
                    # line ??? ??? ?????? ??? or ????????? ???
                    if line[:5] == "-----" or not line:
                        break
                    # ???????????? ????????? ?????? ????????????. (????????? 50??????max -> ????????? ??????)
                    if line[1:len(user_name)+1] == user_name:
                        conversation = line[16:]
                        all_conversation_arr.append(
                            [user_name, cur_time, conversation])

        for i in range(len(all_conversation_arr)):
            all_conversation_arr[i][1] = re.sub(
                "\n", "", all_conversation_arr[i][1])
            all_conversation_arr[i][2] = re.sub(
                "\n", "", all_conversation_arr[i][2])

        # ????????? ?????? ?????? ??????  ???????????? - ?????? ???????????? ??????
        remove_letters = "0123456789???????????????????????????[??????????????????????????????,_???????????????;]'?????????)=(????????????abcdefghijklmnopqrstuvwxyz/QWERTYUIOPASDFGHJKLZXCVBNM#%-\":"
        for i in reversed(range(len(all_conversation_arr))):
            for x in all_conversation_arr[i][2]:
                if x in remove_letters:
                    del all_conversation_arr[i]
                    break

        # ????????? 2 ????????? ????????? ??????
        for i in reversed(range(len(all_conversation_arr))):
            if len(all_conversation_arr[i][2]) <= 10 or len(all_conversation_arr[i][2]) > 56:
                del all_conversation_arr[i]

        all_conversation_arr.reverse()
        # ?????? ?????? 50??? ?????????
        num_50_conversation = all_conversation_arr[:50]
        # ?????? ?????? 100??? ?????????
        num_100_conversation = all_conversation_arr[:100]
        f.close()
        emo_dict = {0: '??????', 1: '??????', 2: '??????', 3: '??????', 4: '??????'}
        result = np.zeros((1, 7), dtype=float)
        for i in all_conversation_arr:
            _sentence = str(i[2])
            if _sentence == '-1':
                break
            transform = nlp.data.BERTSentenceTransform(
                self.tok, max_seq_length=64, pad=True, pair=False)
            sentence = [transform([_sentence])]
            dataloader = torch.utils.data.DataLoader(sentence, batch_size=1)
            _token_ids = dataloader._index_sampler.sampler.data_source

            _t = torch.from_numpy(_token_ids[0][0])
            _t = _t.tolist()
            token_ids = torch.tensor(_t, dtype=torch.long).unsqueeze(0).cuda()
            val_len = torch.tensor([len(token_ids[0])],
                                   dtype=torch.long).cuda()

            _s = torch.from_numpy(_token_ids[0][1])
            _s = _s.tolist()
            segment_ids = torch.tensor(
                _s, dtype=torch.long).unsqueeze(0).cuda()

            out = self.model(token_ids, val_len, segment_ids)
            out_idx = np.argmax(out.cpu().detach().numpy())
            softmax = nn.Softmax(dim=1)
            score = softmax(out).cpu().detach().numpy()
            result += score
        print(emo_dict[np.delete(result, [1, 4]).argmax()])
        print(np.delete(result, [1, 4]))
        print(result.argmax())


# In[ ]:
