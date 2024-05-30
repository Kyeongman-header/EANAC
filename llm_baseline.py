import os
import jsonlines
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm, trange
import numpy as np
import pickle
from seqlearn.hmm import *
from transformers import AutoTokenizer
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import random
encoder = OneHotEncoder(categories=[range(3)], sparse=False)
current_dir = os.path.abspath(__file__)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
def transition(string):
    if string=="intro":
        return 0
    elif string=="body_1":
        return 1
    elif string=="body_2":
        return 2
    elif string=="body_3":
        return 3
    elif string=="ending":
        return 4
def bert_eval(sent_infos,model,train_name_of_labels,test_name_of_labels):
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

    X=[]
    y=[]
    name_y=[]
    A=[]
    lengths=[]
    batch_size=32
    sents=[]
    labels=[]
    for j,sent_info in enumerate(sent_infos):
        sents.append(sent_info['sentence'])
        if sent_info['label']==0 or sent_info['label']==1:
            label=0
        elif sent_info['label']==2:
            label=1
        else:
            label=2
        labels.append(label)
        if (j+1)%batch_size==0:
            tokens=tokenizer(sents,padding='longest',max_length=512,truncation=True,return_tensors="pt") # shape -> (seq_len,word_size)
            attention_mask=tokens['attention_mask']
            tokens=tokens['input_ids']
            X.append(tokens)
            A.append(attention_mask)
            y.append( torch.tensor(encoder.fit_transform(torch.Tensor(labels).view(-1,1)))) # shape -> (seq_len)
            sents=[]
            labels=[]
    
    # model.eval()
    model.train()
    total_eval_accuracy = 0
    predictions , true_labels = [], []

    for batch in zip(tqdm(X),A,y):
        b_input_ids = batch[0].to('cuda')
        b_input_mask = batch[1].to('cuda')
        b_labels = batch[2].to('cuda')
        
        with torch.no_grad():        
            outputs = model(b_input_ids,attention_mask=b_input_mask,)
        
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        
        predictions.append(logits)
        true_labels.append(b_labels.to('cpu').numpy())
        

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = np.argmax(true_labels, axis=1).flatten()
    # input()
    print(np.var(pred_flat))
    print(np.bincount(pred_flat))

    print(classification_report(labels_flat, pred_flat))
    preds=[]
    last_p=0
    for j,p in enumerate(pred_flat):
        if last_p>p and labels_flat[j]!=0:
            p=last_p
        elif labels_flat[j]==0:
            p=0
        
        pred_flat[j]=p
        preds.append({'prediction' : int(p), 'golden_label' : int(labels_flat[j]),'sentence' : sent_infos[j]['sentence']})
        last_p=p
    
    result=classification_report(labels_flat, pred_flat,output_dict=True)
    print(result)
    print(result['accuracy'])
    return preds


def hmm_eval(sent_infos,train_name_of_labels,num_category,HMM):
    X=[]
    y=[]
    name_y=[]
    lengths=[]
    last_label="intro"
    sent_infos=sent_infos[:3000]
    for sent_info in sent_infos:
        if 'label' in sent_info:
            tokens=tokenizer(sent_info['sentence'],return_tensors="np")['input_ids']
            X.append(tokens[0])
            lengths.append(np.shape(tokens[0])[0])
            name_y.append(sent_info['name_label'])

    
    _X=np.concatenate(X)
    I=np.eye(num_category)
    X=I[_X]
    print(np.shape(X))

    _y=HMM.predict(X,lengths)

    sent_class=np.repeat(0,3) # 매번 문장마다, 각 word의 label의 분포를 저장할 배열.

    sent_n=0
    acc=0
    cumul=0
    preds=[]
    last=0
    for j,__y in enumerate(_y):
        if lengths[sent_n]+cumul==j: # 각 문장의 끝에 도달하면, 이때까지 word들의 label 분포에서 argmax를 찾아 그것으로 문장 전체의 label을 매긴다.
            
            pred=np.argmax(sent_class)
            golden = 0 if name_y[sent_n] == "intro" or name_y[sent_n]=="body_1" else ( 1 if name_y[sent_n]=="body_2" else 2)
            # if name_y[sent_n]==pred:
            if last>pred and golden!=0:
                pred=last
            elif golden==0:
                pred=0
            
            if golden==pred:
                acc+=1
            
            
            
            preds.append({"golden_label" : name_y[sent_n], "pred_label" : int(pred), "sentence" : sent_infos[sent_n]['sentence']})
            
            sent_n+=1
            cumul=j
            sent_class=np.repeat(0,3) # 다시 모든 label 0으로 초기화. 
            last=pred

        # print(sent_class)
        sent_class[int(__y)]+=1

    print("total_accuracy : ")
    print(acc/sent_n)
    return preds

import json

if __name__ == "__main__":
    


    with open(os.path.dirname(current_dir) + "/dataset/"+ "llm_test.pkl", 'rb') as f:
        test_sent_infos = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ "llm_test_name_of_labels.pkl", 'rb') as f:
        test_name_of_labels = pickle.load(f)

    with open(os.path.dirname(current_dir) + "/results/wp_rp_bkHMM.pkl", 'rb') as f:
        HMM = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ "wp_rp_bk_train_name_of_labels.pkl", 'rb') as f:
        train_name_of_labels = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/results/"+ "num_category.pkl", 'rb') as f:
        num_category = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/results/wp_rp_bkbert.pkl", 'rb') as f:
        model= pickle.load(f)


    # preds=hmm_eval(test_sent_infos,train_name_of_labels,num_category,HMM)
    # with jsonlines.open(os.path.dirname(current_dir) + "/results/"+"HMM_prediction_results.jsonl", 'w') as fw:
    #     fw.write_all(preds)
    preds=bert_eval(test_sent_infos,model,train_name_of_labels,test_name_of_labels)
    # print(preds)

    with jsonlines.open(os.path.dirname(current_dir) + "/results/"+"llmbert_prediction_results.jsonl", 'w') as fw:
        fw.write_all(preds)
