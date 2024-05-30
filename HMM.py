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
num_category=30522


def train(sent_infos,train_name_of_labels,):
    global num_category
    HMM=MultinomialHMM()
    X=[]
    y=[]
    lengths=[]
    sent_infos=sent_infos[:4000]
    for j,sent_info in enumerate(sent_infos):
        if 'label' in sent_info:
            tokens=tokenizer(sent_info['sentence'],return_tensors="np")['input_ids'] # shape -> (seq_len,word_size)
            X.append(tokens[0])
            if train_name_of_labels[sent_info['label']]=='intro' or train_name_of_labels[sent_info['label']]=='body_1':
                label=0
            elif train_name_of_labels[sent_info['label']]=='body_2':
                label=1
            else:
                label=2
            y.append(np.repeat(label,np.shape(tokens[0])[0])) # shape -> (seq_len)
            lengths.append(np.shape(tokens[0])[0]) # [seq_len_1, seq_len_2, seq_len_3, ...]
        
    _X=np.concatenate(X)
    y=np.concatenate(y)
    # num_category = np.max(_X)+1
    I=np.eye(num_category)
    X=I[_X]
    print(np.shape(X)) # it should be (sum_of_seqences_N, word_size)
    print(np.shape(y)) # it should be (sum_of_seqences_N,)
    
    HMM=HMM.fit(X,y,lengths)
    return HMM, num_category

def eval(sent_infos,HMM,train_name_of_labels,test_name_of_labels):
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
    for j,__y in enumerate(_y):
        if lengths[sent_n]+cumul==j: # 각 문장의 끝에 도달하면, 이때까지 word들의 label 분포에서 argmax를 찾아 그것으로 문장 전체의 label을 매긴다.
            
            pred=np.argmax(sent_class)
            golden = 0 if name_y[sent_n] == "intro" or name_y[sent_n]=="body_1" else ( 1 if name_y[sent_n]=="body_2" else 2)
            # if name_y[sent_n]==pred:
            if golden==pred:
                acc+=1
            
            preds.append({"golden_label" : name_y[sent_n], "pred_label" : int(pred), "sentence" : sent_infos[sent_n]['sentence']})
            sent_n+=1
            cumul=j
            sent_class=np.repeat(0,3) # 다시 모든 label 0으로 초기화. 

        # print(sent_class)
        sent_class[int(__y)]+=1

    print("total_accuracy : ")
    print(acc/sent_n)
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="writingPrompts_reedsyPrompts_booksum",action="store")
    
    args = parser.parse_args()

    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_train.pkl", 'rb') as f:
        train_sent_infos = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_train_name_of_labels.pkl", 'rb') as f:
        train_name_of_labels = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ "llm_test.pkl", 'rb') as f:
        test_sent_infos = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ "llm_test_name_of_labels.pkl", 'rb') as f:
        test_name_of_labels = pickle.load(f)
    
    HMM,num_category=train(train_sent_infos,train_name_of_labels)
    preds=eval(test_sent_infos,HMM,train_name_of_labels,test_name_of_labels)
    
    # print(preds)

    with jsonlines.open(os.path.dirname(current_dir) + "/results/"+args.dataset+"_prediction_results.jsonl", 'w') as fw:
        fw.write_all(preds)

    with open(os.path.dirname(current_dir) + "/results/"+args.dataset+"HMM.pkl", 'wb') as f:
        pickle.dump(HMM, f)
    
    with open(os.path.dirname(current_dir) + "/results/"+ "num_category.pkl", 'wb') as f:
        pickle.dump(num_category, f)




