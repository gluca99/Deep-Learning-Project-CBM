import os
import json
import openai
import time

import data_utils
import argparse
parser = argparse.ArgumentParser(description='Settings for creating CBM')


openai.api_key = open(os.path.join("~/concept-driven-continual-learning/sandbox-lf-cbm/", ".openai_api_key"), "r").read()[:-1]

prompts = {
    "important" : "List the most important features for recognizing something as a \"goldfish\":\n\n-bright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\nList the most important features for recognizing something as a \"beerglass\":\n\n-a tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\nList the most important features for recognizing something as a \"{}\":",
    "superclass" : "Give superclasses for the word \"tench\":\n\n-fish\n-vertebrate\n-animal\n\nGive superclasses for the word \"beer glass\":\n\n-glass\n-container\n-object\n\nGive superclasses for the word \"{}\":",
    "around" : "List the things most commonly seen around a \"tench\":\n\n- a pond\n-fish\n-a net\n-a rod\n-a reel\n-a hook\n-bait\n\nList the things most commonly seen around a \"beer glass\":\n\n- beer\n-a bar\n-a coaster\n-a napkin\n-a straw\n-a lime\n-a person\n\nList the things most commonly seen around a \"{}\":"
}
parser.add_argument("--dataset", type=str, default="cifar100")

def get_init(dataset,prompt_type,base_prompt):
    cls_file="~/concept-driven-continual-learning/sandbox-lf-cbm/data/%s_classes.txt" % (dataset)
    #cls_file = data_utils.LABEL_FILES[dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")


    feature_dict = {}

    for i, label in enumerate(classes):
        if(i%25==0):
            time.sleep(60)
        feature_dict[label] = set()
        for _ in range(2):
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=base_prompt.format(label),
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
            #clean up responses
            features = response["choices"][0]["text"]
            features = features.split("\n-")
            features = [feat.replace("\n", "") for feat in features]
            features = [feat.strip() for feat in features]
            features = [feat for feat in features if len(feat)>0]
            features = set(features)
            feature_dict[label].update(features)
        feature_dict[label] = sorted(list(feature_dict[label]))


    json_object = json.dumps(feature_dict, indent=4)
    with open("~/concept-driven-continual-learning/sandbox-lf-cbm/data/concept_sets/gpt3_init/gpt3_{}_{}.json".format(dataset, prompt_type), "w") as outfile:
        outfile.write(json_object)

if __name__=='__main__':
    args = parser.parse_args()
    seed_list=['3456']
    task_num=20
    for s in seed_list:
        print('********* SEED:', s)
        for t in range(task_num):
            print("Working on task: ",t)
            dataset = "%s_task_%i_%i_%s" % (args.dataset,t,task_num,s)
            for prompt_type in prompts.keys():
                base_prompt = prompts[prompt_type]
                get_init(dataset,prompt_type,base_prompt)

