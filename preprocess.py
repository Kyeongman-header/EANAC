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

current_dir = os.path.abspath(__file__)

def plot(dataset,X,labels,n_clusters_,db,eps,avg_pos_classes):
    plt.clf()
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        label="label " + str(k) + " avg pos " + str(avg_pos_classes[k])
        print(label)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14,label=label)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6, )


    plt.legend(loc="upper left")
    
    plt.title(dataset + ' Estimated number of clusters: %d' % n_clusters_)
    
    plt.savefig(dataset + "_eps_"+str(eps)+"_clusters_"+str(n_clusters_)+".png")


def sentence_clustering(dataset,stories):
    
    # step 1. sentence emebdding.
    sentence_bert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",device="cuda")
    vectors=[]
    sent_infos=[]

    #stories=stories[:10]

    for story in tqdm(stories):
        sents=sent_tokenize((' ').join(story))
        for j,sent in enumerate(sents):
            
            sent_vec={"emb" : sentence_bert.encode(sent),"pos":j/len(sents),"sentence" : sent}
            vectors.append(sent_vec["emb"])
            sent_infos.append(sent_vec)

    vectors=np.array(vectors)
    # step 2. clustering by DBScan.
    print("huristic search for hyperparameter.")
    while True:
        print("epsilon?")
        eps=input()
        # if eps.isdigit() is False:
        #     print("input a real number.")
        #     continue
        eps=float(eps)
        clustering = DBSCAN(eps=eps, min_samples=10,metric='cosine',n_jobs=-1).fit(vectors)
        
        labels=clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

        print("(close to 1 = best) Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(vectors, labels))
        


        avg_pos_classes=np.zeros(n_clusters_, dtype=float)
        num_pos_classes=np.zeros(n_clusters_, dtype=float)

        for j in range(len(sent_infos)):
            avg_pos_classes[labels[j]]+=sent_infos[j]['pos']
            num_pos_classes[labels[j]]+=1
        
        print(avg_pos_classes)
        print(num_pos_classes)

        for j in range(len(avg_pos_classes)):
            avg_pos_classes[j]=avg_pos_classes[j]/num_pos_classes[j]
        
        plot(dataset,vectors,labels,n_clusters_,clustering,eps,avg_pos_classes)


        print("input the name of label")
        name_of_labels=[]
        for label in range(n_clusters_):
            name_of_labels.append(input(str(label) + " : "))
        print("if clustering done, enter 'x'. otherwise, it iterate.")
        x=input()
        if x == "x":
            break
    
    for j,label in enumerate(labels):
        sent_infos[j]['name_label']=name_of_labels[label]
        sent_infos[j]['label']=label
        del sent_infos[j]['emb']

    

    return sent_infos
    # step 3. 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", default="writingPrompts_reedsyPrompts_booksum",action="store")
    
    args = parser.parse_args()

    
    stories=[]
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", args.dataset + "_train_points.jsonl")) as f:
        for line in f:
            stories.append(line['stories'])

    print(len(stories))

    train_sent_infos=sentence_clustering("train",stories)

    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_train.pkl", 'wb') as f:
        pickle.dump(train_sent_infos, f,)

    stories=[]
    with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", args.dataset + "_test_points.jsonl")) as f:
        for line in f:
            stories.append(line['stories'])
    
    print(len(stories))
    test_sent_infos=sentence_clustering("test",stories)

    # with jsonlines.open(os.path.dirname(current_dir) + "/results/"+args.file_dir+"/evaluation_results.jsonl", mode) as fw:
    #     fw.write_all(result)
    with open(os.path.dirname(current_dir) + "/dataset/"+ args.dataset + "_test.pkl", 'wb') as f:
        pickle.dump(test_sent_infos, f,)
