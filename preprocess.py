import os
import argparse
import jsonlines
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm, trange
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from transformers import AutoTokenizer
import torch
from sklearnex import patch_sklearn, config_context
from sklearn.cluster import KMeans
import time
patch_sklearn()
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

current_dir = os.path.abspath(__file__)


def sentence_clustering(dataset,stories,start):
    
    # step 1. sentence emebdding.
    sentence_bert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",device="cuda")
    vectors=[]
    sent_infos=[]

    # stories=stories[:10]

    whole_sents=[]
    count=0
    for story in tqdm(stories):
        sents=sent_tokenize((' ').join(story))
        whole_sents.extend(sents)
        for j,sent in enumerate(sents):
        
            # emb=tokenizer(sent,return_tensors="np")['input_ids']
            # emb=np.sum(emb,axis=0)
            sent_vec={"pos":j/len(sents),"sentence" : sent, 'story_label' : count}
            sent_infos.append(sent_vec)
        count+=1
    
    # print(whole_sents)
    len_whole=len(whole_sents)
    print(len_whole)

    vectors=sentence_bert.encode(whole_sents,batch_size=64, show_progress_bar=True)
    
    # step 2. clustering by DBScan.
    num_clusters = 5
    clustering_model = KMeans(n_clusters=num_clusters)
    print("KMeans operation start.")
    start = time.time()
    with config_context(target_offload="gpu:0"):
        clustering_model.fit(vectors)
    end = time.time()
    print(f"{end - start:.5f} sec")
    print("KMeans operation ends.")
    
    
    cluster_assignment = clustering_model.labels_

    avg_pos_classes=np.zeros(5, dtype=float)
    num_pos_classes=np.zeros(5, dtype=float)

    for sentence_id, cluster_id in enumerate(cluster_assignment):
        avg_pos_classes[cluster_id]+=sent_infos[sentence_id]['pos']
        num_pos_classes[cluster_id]+=1


    print("number of each class")
    print(num_pos_classes)
    
    for j in range(len(avg_pos_classes)):
        avg_pos_classes[j]=avg_pos_classes[j]/num_pos_classes[j]
    
    print("avg position of classes")
    print(avg_pos_classes)
    

    sorted_avg_pos_classes=sorted(avg_pos_classes)
    
    print("input the name of label")
    name_of_labels=[]

    name_rank=['intro','body_1','body_2','body_3','ending']

    for label in range(5):
        rank=sorted_avg_pos_classes.index(avg_pos_classes[label])
        name_of_labels.append(name_rank[rank])

    print(name_of_labels)

    for sentence_id, cluster_id in enumerate(cluster_assignment):
        sent_infos[sentence_id]['label']=cluster_id
        sent_infos[sentence_id]['name_label']=name_of_labels[cluster_id]



    # print("huristic search for hyperparameter.")

    # count=0
    # while True:
    #     print("min_community_size?")
    #     # min_community_size=input()
    #     min_community_size=start+count*100
    #     print(min_community_size)

    #     clusters = util.community_detection(vectors, min_community_size=int(min_community_size), threshold=0.75,)
    #     avg_pos_classes=np.zeros(len(clusters), dtype=float)
    #     num_pos_classes=np.zeros(len(clusters), dtype=float)


    #     print(len(clusters))
    #     if len(clusters)>5:
            
    #         print("too many clusters.")
    #         count+=1
    #         continue
    #     elif len(clusters)<5:
    #         print("too low clusters.")
    #         count-=0.2
    #         continue
        
    #     for i, cluster in enumerate(clusters):
        
    #         print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))

    #         for sentence_id in cluster:
    #             avg_pos_classes[i]+=sent_infos[sentence_id]['pos']
    #             num_pos_classes[i]+=1


    #     print("number of each class")
    #     print(num_pos_classes)
        
    #     for j in range(len(avg_pos_classes)):
    #         avg_pos_classes[j]=avg_pos_classes[j]/num_pos_classes[j]
        
    #     print("avg position of classes")
    #     print(avg_pos_classes)
        

    #     sorted_avg_pos_classes=sorted(avg_pos_classes)
        
    #     print("input the name of label")
    #     name_of_labels=[]

    #     name_rank=['intro','body_1','body_2','body_3','ending']

    #     for label in range(len(clusters)):
    #         rank=sorted_avg_pos_classes.index(avg_pos_classes[label])
    #         name_of_labels.append(name_rank[rank])

    #     print(name_of_labels)

    #     sentence_ids=[]
    #     for i, cluster in enumerate(clusters):
    #         for sentence_id in cluster:
    #             sentence_ids.append(sentence_id)
    #             # print(sentence_id)
    #             # print(sent_infos[sentence_id])

    #             sent_infos[sentence_id]['name_label']=name_of_labels[i]
    #             sent_infos[sentence_id]['label']=i
    #             # print(sent_infos[sentence_id])
        
    #     for j,sent_info in enumerate(sent_infos):
    #         if 'label' in sent_info is False:
    #             if last_story != sent_info['story_label']:
    #                 sent_info['name_label']='intro'
    #                 sent_info['label']=name_of_labels.index('intro')
    #             else :
    #                 sent_info['name_label']=last_label
    #                 sent_info['label']=name_of_labels.index(last_label)
            
    #         last_label=sent_info

    #     break
    

    return sent_infos,name_of_labels

import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="wp_rp_bk",action="store")
    
    args = parser.parse_args()

    stories=[]
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", args.dataset + "_test.jsonl")) as f:
        for line in f:
            stories.append(line['stories'])
    
    print(len(stories))
    stories=random.sample(stories, k=3000)
    test_sent_infos,name_of_labels=sentence_clustering("test",stories,100)



    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_test.pkl", 'wb') as f:
        pickle.dump(test_sent_infos, f,)
    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_test_name_of_labels.pkl", 'wb') as f:
        pickle.dump(name_of_labels, f,)
    
    stories=[]
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", args.dataset + "_train.jsonl")) as f:
        for line in f:
            stories.append(line['stories'])

    print(len(stories))
    stories=random.sample(stories, k=30000)

    train_sent_infos,name_of_labels=sentence_clustering("train",stories,960)

    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_train.pkl", 'wb') as f:
        pickle.dump(train_sent_infos, f,)
    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_train_name_of_labels.pkl", 'wb') as f:
        pickle.dump(name_of_labels, f,)