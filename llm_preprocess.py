import json
import pickle
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm, trange
import jsonlines
current_dir = os.path.abspath(__file__)
def maker(llm_data,name):
    stories=[]
    name_labels=[]
    repeat_name_label=[]
    print(llm_data)
    error_count=0
    for (path, d, files) in os.walk(llm_data):
        for file in files:

            print(llm_data+"/"+file)
            with open(llm_data+"/"+file, 'r', encoding='utf-8') as f:
                try:
                    data=json.load(f)
                except:
                    error_count+=1
                    continue
            #     data=f.read()
            #     # print(data)
            #     data=data.replace('```json','').replace('```','').replace(',\n]',']').replace('}}','}\n]}').replace("\"","``").replace("``story``","\"story\"").replace("``label``","\"label\"").replace("``sentence``","\"sentence\"")
            #     data=data.replace("``introduction``","\"introduction\"").replace("``rising action``","\"rising action\"").replace("``conflict``","\"conflict\"").replace("``climax``","\"climax\"").replace("``resolution``","\"resolution\"").replace(": ``",": \"").replace("``}","\"}")
            #     data=data.replace("Introduction``","introduction\"").replace("Rising Action``","rising action\"").replace("Conflict``","conflict\"").replace("Climax``","climax\"").replace("Resolution``","resolution\"")
                
            #     print(data)
                
            # with open(llm_data+"/"+file, 'w', encoding='utf-8') as f:
            #     f.write(data)
            if ('story' in data) is False:
                continue
            
            with open(llm_data+"/"+file, 'r', encoding='utf-8') as f:
                try:
                    data=json.load(f)
                except:
                    error_count+=1
                    continue
            for sent in data['story']:
                sents=sent_tokenize(sent['sentence'])
                stories+=sents
                if sent['label']=='introduction':
                    name_label='intro'
                elif sent['label']=='rising action':
                    name_label='body_1'
                elif sent['label']=='conflict':
                    name_label='body_2'
                elif sent['label']=='climax':
                    name_label='body_3'
                elif sent['label']=='resolution':
                    name_label='ending'
                repeat_name_label = [name_label] * len(sents)
                name_labels+=repeat_name_label

    print(error_count)
    print(len(stories))
    print(len(name_labels))
    print(stories[:3])
    print(name_labels[:3])

    sent_infos=[]
    for j,(sent,label) in enumerate(zip(tqdm(stories),name_labels)):
    
        # emb=tokenizer(sent,return_tensors="np")['input_ids']
        # emb=np.sum(emb,axis=0)
        sent_vec={"pos":j/len(sents),"sentence" : sent,"name_label":label,"label":name_of_labels.index(label)}
        sent_infos.append(sent_vec)
    
    with open(os.path.dirname(current_dir) + "/dataset/llm_" + name + ".pkl", 'wb') as f:
        pickle.dump(sent_infos, f,)
    with open(os.path.dirname(current_dir) + "/dataset/llm_" + name+ "_name_of_labels.pkl", 'wb') as f:
        pickle.dump(name_of_labels, f,)


if __name__ == "__main__":
    name_of_labels=['intro','body_1','body_2','body_3','ending']


    llm_data=os.path.join(os.path.dirname(current_dir), "stories_GPT3_5-turbo")
    # 0~4 중 무엇이 INTRO, RISING ACTION 등등인지 알 수 있게 해야함.
    maker(llm_data,"test")
    llm_data=os.path.join(os.path.dirname(current_dir), "my_GPT3_5_stories")
    maker(llm_data,"train")
    
    