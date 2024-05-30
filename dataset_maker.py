import jsonlines
import os
from tqdm import tqdm, trange
import pickle
import math
import unicodedata
current_dir = os.path.abspath(__file__)

T="test"

train_data=os.path.join(os.path.dirname(current_dir), "dataset/writingPrompts", "train.wp_target")
valid_data=os.path.join(os.path.dirname(current_dir), "dataset/writingPrompts","valid.wp_target")
test_data=os.path.join(os.path.dirname(current_dir), "dataset/writingPrompts","test.wp_target")
data=[test_data,]
# data=[valid_data,test_data,train_data]
for k,name in enumerate(data):
    with open(name) as f:
        stories = f.readlines()

        # if k!=2:
        #    stories = stories[:few_number // 10]
        # else:
        #    stories = stories[:few_number]

        # stories = stories[847:few_number]
        
        print(len(stories))
        
        
        for story in tqdm(stories):
            s=[{'stories':[story]}]
            if os.path.exists(os.path.dirname(current_dir)+"/dataset/writingPrompts_"+T+".jsonl"):
                mode="a"
            else:
                mode="w"

            with jsonlines.open(os.path.dirname(current_dir)+"/dataset/writingPrompts_"+T+".jsonl", mode=mode) as f:
                f.write_all(s)
            
            if os.path.exists(os.path.dirname(current_dir)+"/dataset/wp_rp_bk_"+T+".jsonl"):
                mode="a"
            else:
                mode="w"

            with jsonlines.open(os.path.dirname(current_dir)+"/dataset/wp_rp_bk_"+T+".jsonl", mode=mode) as f:
                f.write_all(s)



train_data=os.path.join(os.path.dirname(current_dir), "dataset/reedsyPrompts", "reedsy_prompts_whole.pickle")
with open(train_data,"rb") as fi:
    reedsy = pickle.load(fi)


reedsy_len=len(reedsy)        
test_len=math.ceil(reedsy_len*0.1)
valid_len=test_len*2
print(reedsy_len)

for k,line in enumerate(tqdm(reedsy[-test_len:])):
    story=unicodedata.normalize("NFKD", line['story'])
    s=[{'stories':[story]}]
    
    if os.path.exists(os.path.dirname(current_dir)+"/dataset/reedsyPrompts_"+T+".jsonl"):
        mode="a"
    else:
        mode="w"

    with jsonlines.open(os.path.dirname(current_dir)+"/dataset/reedsyPrompts_"+T+".jsonl", mode=mode) as f:
        f.write_all(s)
    
    if os.path.exists(os.path.dirname(current_dir)+"/dataset/wp_rp_bk_"+T+".jsonl"):
        mode="a"
    else:
        mode="w"

    with jsonlines.open(os.path.dirname(current_dir)+"/dataset/wp_rp_bk_"+T+".jsonl", mode=mode) as f:
        f.write_all(s)
    
train_data=os.path.join(os.path.dirname(current_dir), "dataset/booksum")



for (path, d, files) in os.walk(train_data):
    booksum_len=len(d)
    
    
    test_len=math.ceil(booksum_len*0.1)
    valid_len=test_len*2
    

    print(booksum_len)
    # input()

    for k,b in enumerate(tqdm(d[-test_len:])):
        story_file=path+'/'+b + '/book_clean.txt'
        story=""
        with open(story_file,"r") as f:
            story=f.readlines()
        story=' '.join(story)
        
        s=[{'stories':[story]}]
        if os.path.exists(os.path.dirname(current_dir)+"/dataset/booksum_"+T+".jsonl"):
            mode="a"
        else:
            mode="w"

        with jsonlines.open(os.path.dirname(current_dir)+"/dataset/booksum_"+T+".jsonl", mode=mode) as f:
            f.write_all(s)
        
        if os.path.exists(os.path.dirname(current_dir)+"/dataset/wp_rp_bk_"+T+".jsonl"):
            mode="a"
        else:
            mode="w"

        with jsonlines.open(os.path.dirname(current_dir)+"/dataset/wp_rp_bk_"+T+".jsonl", mode=mode) as f:
            f.write_all(s)

    break
