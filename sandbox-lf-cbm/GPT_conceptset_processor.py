
import json
import data_utils
import conceptset_utils
import argparse

parser = argparse.ArgumentParser(description='Settings for creating CBM')
parser.add_argument("--dataset", type=str, default="cifar100")

"""
CLASS_SIM_CUTOFF: Concenpts with cos similarity higher than this to any class will be removed
OTHER_SIM_CUTOFF: Concenpts with cos similarity higher than this to another concept will be removed
MAX_LEN: max number of characters in a concept

PRINT_PROB: what percentage of filtered concepts will be printed
"""

CLASS_SIM_CUTOFF = 0.85
OTHER_SIM_CUTOFF = 0.9
MAX_LEN = 30
PRINT_PROB = 1

device = "cuda"

def conceptset(dataset,pre_concept):
    save_name = "~/concept-driven-continual-learning/sandbox-lf-cbm/data/concept_sets/{}_filtered.txt".format(dataset)

    #EDIT these to use the initial concept sets you want

    with open("~/concept-driven-continual-learning/sandbox-lf-cbm/data/concept_sets/gpt3_init/gpt3_{}_important.json".format(dataset), "r") as f:
        important_dict = json.load(f)
    with open("~/concept-driven-continual-learning/sandbox-lf-cbm/data/concept_sets/gpt3_init/gpt3_{}_superclass.json".format(dataset), "r") as f:
        superclass_dict = json.load(f)
    with open("~/concept-driven-continual-learning/sandbox-lf-cbm/data/concept_sets/gpt3_init/gpt3_{}_around.json".format(dataset), "r") as f:
        around_dict = json.load(f)

    cls_file="~/concept-driven-continual-learning/sandbox-lf-cbm/data/%s_classes.txt" % (dataset)
    #cls_file = data_utils.LABEL_FILES[dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    concepts = set()

    for values in important_dict.values():
        concepts.update(set(values))

    for values in superclass_dict.values():
        concepts.update(set(values))
        
    for values in around_dict.values():
        concepts.update(set(values))

    print(len(concepts))

    concepts = conceptset_utils.remove_too_long(concepts, MAX_LEN, PRINT_PROB)

    concepts = conceptset_utils.filter_too_similar_to_cls(concepts, classes, CLASS_SIM_CUTOFF, device, PRINT_PROB)

    concepts = conceptset_utils.filter_too_similar(concepts, OTHER_SIM_CUTOFF, device, PRINT_PROB)

    concepts=list(concepts)
    if (pre_concept!=None):
        concepts=pre_concept+concepts

    with open(save_name, "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)
    
    return concepts

if __name__=='__main__':
    args = parser.parse_args()
    seed_list=['3456']
    task_num=20
    for s in seed_list:
        print('********* SEED:', s)
        pre_concept=None
        for t in range(task_num):
            print("Working on task: ",t)
            dataset = "%s_task_%i_%i_%s" % (args.dataset,t,task_num,s)
            pre_concept = conceptset(dataset,pre_concept)
