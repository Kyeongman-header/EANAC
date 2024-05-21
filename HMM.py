import os
import jsonlines
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm, trange
import numpy as np
import pickle
from seqlearn.hmm import *
from transformers import AutoTokenizer

current_dir = os.path.abspath(__file__)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")



def train(sent_infos):
    HMM=MultinomialHMM()
    X=[]
    y=[]
    lengths=[]
    for sent_info in sent_infos:
        tokens=tokenizer(sent_info['sentence'],return_tensors="np") # shape -> (seq_len,word_size)
        X.append(tokens)
        y.append(np.repeat(sent_info['label'],np.shape(tokens)[0])) # shape -> (seq_len)
        lengths.append(np.shape(tokens)[0]) # [seq_len_1, seq_len_2, seq_len_3, ...] 
    
    X=np.vstack(X)[0]
    y=np.vstack(y)[0]
    print(np.shape(X)) # it should be (sum_of_seqences_N, word_size)
    print(np.shape(y)) # it should be (sum_of_seqences_N,)
    HMM=HMM.fit(X,y,lengths)
    return HMM

def eval(sent_infos,HMM):
    X=[]
    y=[]
    for sent_info in sent_infos:
        tokens=tokenizer(sent_info['sentence'],return_tensors="np")
        X.append(tokens)
        y.append(sent_info['label']) # train 때와 달리, word 단위 별로 labeling을 하지 않아도 된다.
        lengths.append(np.shape(tokens)[0])
    
    X=np.vstack(X)[0]
    print(np.shape(X))

    _y=HMM.predict(X,lengths)

    classes=np.unique(y)
    sent_class=np.repeat(0,len(classes)) # 매번 문장마다, 각 word의 label의 분포를 저장할 배열.

    sent_n=0
    acc=0
    cumul=0
    preds=[]
    for j,__y in enumerate(_y):
        if lengths[sent_n]+cumul==j: # 각 문장의 끝에 도달하면, 이때까지 word들의 label 분포에서 argmax를 찾아 그것으로 문장 전체의 label을 매긴다.
            
            pred=np.argmax(sent_class)
            if y[sent_n]==pred:
                acc+=1
            preds.append({"pred_label" : pred, "sentence" : sent_infos[sent_n]['sentence']})
            sent_n+=1
            cumul=j
            sent_class=np.repeat(0,len(classes)) # 다시 모든 label 0으로 초기화. 

        
        sent_class[str[__y]]+=1

    print("total_accuracy : ")
    print(acc/sent_n)
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="writingPrompts_reedsyPrompts_booksum",action="store")
    
    args = parser.parse_args()

    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_train.pkl", 'rb') as f:
        train_sent_infos = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_test.pkl", 'rb') as f:
        test_sent_infos = pickle.load(f)

    HMM=train(train_sent_infos)
    preds=eval(test_sent_infos,HMM)

    with jsonlines.open(os.path.dirname(current_dir) + "/results/"+args.test_dataset+"_prediction_results.jsonl", mode) as fw:
        fw.write_all(preds)




