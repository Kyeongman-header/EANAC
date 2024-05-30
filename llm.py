import os
import time
from tqdm import tqdm
import jsonlines

from openai import OpenAI

client = OpenAI()
model = "gpt-3.5-turbo"

query = "Please split the following story into appropriate five narrative arc steps: introduction, rising action, conflict, climax and resolution, and label it in a form of list of dictionaries as [{\"label\": \"(narrative arc step)\", \"sentence\": \"~~~\"}, ...].\n"
query += "Each sentence should be recognized as a string in python. I want to save your response as a json directly. So please make the output fit to the format well.\n"
query += "In addition, do not write any character outside of the specified format of [] such as ```json"


current_dir = os.path.abspath(__file__)

stories=[]
with jsonlines.open(os.path.join(os.path.dirname(current_dir), "dataset", "wp_rp_bk_train.jsonl")) as f:
    for line in f:
        stories.append(line['stories'][0])

stories=stories[:1000]

for i in tqdm(range(len(stories)), desc = 'story splitting with GPT: '):
    time.sleep(0.5)
    messages = [
        {"role": "system", "content": query},
        {"role": "user", "content": stories[i] }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    answer = "{ \"story\" : " + response.choices[0].message.content + "}"
    with open("./my_GPT3_5_stories/story_{}.json".format(i),'w', encoding='utf-8') as f:
        f.write(answer)
    f.close()

