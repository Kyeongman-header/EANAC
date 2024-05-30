import os
import jsonlines
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm, trange
import numpy as np
import pickle
from seqlearn.hmm import *
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import random


encoder = OneHotEncoder(categories=[range(3)], sparse=False)
current_dir = os.path.abspath(__file__)
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
num_category=30522

# 에폭 수와 학습률 스케줄러 설정
def create_random_list(original, new_length):
    # 결과 리스트를 빈 리스트로 초기화
    result = []
    
    # 원하는 길이가 될 때까지 반복
    while len(result) < new_length:
        # original 리스트에서 랜덤하게 하나의 요소를 선택하여 추가
        result.append(random.choice(original))
    
    return result

    
def train(model,sent_infos,test_sent_infos,train_name_of_labels,test_name_of_labels):
    X=[]
    y=[]
    A=[]
    lengths=[]
    sent_infos=sent_infos[:100000]

    batch_size=32
    sents=[]
    labels=[]
    z_c=0
    o_c=0
    t_c=0
    z_sents=[]
    o_sents=[]
    t_sents=[]
    for j,sent_info in enumerate(tqdm(sent_infos)):
        if sent_info['label']==0 or sent_info['label']==1 :
            z_sents.append({"sentence" : sent_info['sentence'],"label" : 0})
            z_c+=1
        elif  sent_info['label']==2:
            o_sents.append({"sentence" : sent_info['sentence'],"label" : 1})
            o_c+=1
        else:
            t_sents.append({"sentence" : sent_info['sentence'],"label" : 2})
            t_c+=1
    
    max_len=max(len(z_sents),len(o_sents),len(t_sents))
    print(z_c)  
    print(o_c)
    print(t_c)
    z_sents=create_random_list(z_sents,max_len)
    o_sents=create_random_list(o_sents,max_len)
    t_sents=create_random_list(t_sents,max_len)
    print("after resample")
    print(len(z_sents))
    print(len(o_sents))
    print(len(t_sents))
    whole_sents=z_sents+o_sents+t_sents
    random.shuffle(whole_sents)
    sents=[]
    labels=[]
    for j,sent_info in enumerate(tqdm(whole_sents)):
        sents.append(sent_info["sentence"])
        labels.append(sent_info['label'])
        if (j+1)%batch_size==0:
            tokens=tokenizer(sents,padding='longest',max_length=512,truncation=True,return_tensors="pt") # shape -> (seq_len,word_size)
            attention_mask=tokens['attention_mask']
            tokens=tokens['input_ids']
            X.append(tokens)
            A.append(attention_mask)
            y.append( torch.tensor(encoder.fit_transform(torch.Tensor(labels).view(-1,1)))) # shape -> (seq_len)
            
            sents=[]
            labels=[]

    
    optimizer = AdamW(model.parameters(),
                  lr = 2e-3,  # 학습률
                  eps = 1e-8  # 0으로 나누기 방지를 위한 epsilon 값
                )

    epochs = 10
    total_steps = len(X) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

    last_avg_token_loss=9999
    over_count=0
    last_result=0
    for epoch_i in range(epochs):
        model.train()
        total_train_loss = 0
        steps = 0
        predictions=[]
        
        for step, batch in enumerate(zip(tqdm(X),A,y)):
            b_input_ids = batch[0].to('cuda')
            b_input_mask = batch[1].to('cuda')
            b_labels = batch[2].to('cuda')
            
            model.zero_grad()
            
            outputs = model(b_input_ids, 
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
            loss = outputs.loss
            logits = outputs.logits.detach().cpu().numpy()
        
            predictions.append(logits)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그래디언트 클리핑
            optimizer.step()
            scheduler.step()
            steps += 1
            

            if steps % 150 == 0:

                avg_train_loss = total_train_loss / steps
                
                print(f"Epoch {epoch_i} Batch {steps} / {len(X)}, Average Training Loss: {avg_train_loss}")
                if last_avg_token_loss<avg_train_loss:
                    over_count+=1
                    if over_count>3:
                        return model, num_category
                else:
                    over_count-=1
                    if over_count<0:
                        over_count=0
                last_avg_token_loss=avg_train_loss

                predictions = np.concatenate(predictions, axis=0)
                pred_flat = np.argmax(predictions, axis=1).flatten()
                
                print(pred_flat)
                print(np.var(pred_flat))
                print(np.bincount(pred_flat))
                predictions=[]
                
        if epoch_i>0:
            preds,result=eval(test_sent_infos,model,train_name_of_labels,test_name_of_labels)
            print("eval result")
            print(result)
            if result<last_result:
                
                return last_model, num_category
            last_result=result
            last_model=model        
                
    
    return model, num_category
    
def eval(sent_infos,model,train_name_of_labels,test_name_of_labels):
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
    print(np.bincount(pred_flat))
    # input()
    print(np.var(pred_flat))

    
    print(classification_report(labels_flat, pred_flat))
    preds=[]
    for j,p in enumerate(pred_flat):
        preds.append({'prediction' : int(p), 'golden_label' : int(labels_flat[j]),'sentence' : sent_infos[j]['sentence']})
    
    result=classification_report(labels_flat, pred_flat,output_dict=True)
    # print(result)
    # print(result['accuracy'])
    return preds, result['accuracy']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="writingPrompts_reedsyPrompts_booksum",action="store")
    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity",num_labels=3,ignore_mismatched_sizes=True).to('cuda')

    args = parser.parse_args()

    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_train.pkl", 'rb') as f:
        train_sent_infos = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_train_name_of_labels.pkl", 'rb') as f:
        train_name_of_labels = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ "llm_test.pkl", 'rb') as f:
        test_sent_infos = pickle.load(f)
    with open(os.path.dirname(current_dir) + "/dataset/"+ "llm_test_name_of_labels.pkl", 'rb') as f:
        test_name_of_labels = pickle.load(f)
    
    
    
    
    # with open(os.path.dirname(current_dir) + "/results/"+ "bert.pkl", 'rb') as f:
    #     model=pickle.load(f)
    
    model,num_category=train(model,train_sent_infos,test_sent_infos,train_name_of_labels,test_name_of_labels)
    
    preds,result=eval(test_sent_infos,model,train_name_of_labels,test_name_of_labels)
    
    # print(preds)

    with jsonlines.open(os.path.dirname(current_dir) + "/results/"+args.dataset+"bert_prediction_results.jsonl", 'w') as fw:
        fw.write_all(preds)

    with open(os.path.dirname(current_dir) + "/results/"+ args.dataset+ "bert.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.dirname(current_dir) + "/results/"+ "num_category.pkl", 'wb') as f:
        pickle.dump(num_category, f)




