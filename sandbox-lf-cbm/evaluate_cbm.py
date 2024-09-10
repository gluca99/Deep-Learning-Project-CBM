# %%
import torch
import os
import data_utils
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import cbm

device = "cuda"
parser = argparse.ArgumentParser(description='Settings for creating CBM')

parser.add_argument("--dataset", type=str, default="cifar100")
parser.add_argument("--seed", type=str, default="3456")
parser.add_argument("--save_dir", type=str, default="save_dir")
parser.add_argument("--task_num", type=int, default=10)
parser.add_argument("--nonlinear", type=str, default="False")
# %%
# change this to the correct model dir, everything else should be taken care of
def get_dataset(load_dir,seed):
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
    dataset = args["dataset"]
    _, target_preprocess = data_utils.get_target_model(args["backbone"],dataset,seed,args["train1"], device)
    val_d_probe = dataset+"_val"
    _, val_data_t = data_utils.get_data(val_d_probe,seed, clip_preprocess=target_preprocess, target_preprocess=target_preprocess)
    return val_data_t,dataset,args["train1"]

# %%
def get_model(model_dir,dataset,seed,train1,nonlinear):
    model = cbm.load_cbm(model_dir,dataset,seed,train1,nonlinear, device)
    return model


def eval(val_data_t,model):
    correct = 0
    total = 0
    for images, labels in tqdm(DataLoader(val_data_t, 50, num_workers=2, pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    print("Accuracy: {:.2f}%".format(correct/total*100))
    return (correct/total)

if __name__=='__main__':
    args = parser.parse_args()
    result_list={}
    for d in range(args.task_num):
        result_list[d]=[]
        load_dir='%s/%s_task_%d_%d_%s_cbm' % (args.save_dir,args.dataset,d,args.task_num,args.seed)
        print("load data:",load_dir)
        val_data,dataset,train1=get_dataset(load_dir,args.seed)
        for m in range(d,args.task_num):
            model_dir='%s/%s_task_%d_%d_%s_cbm' % (args.save_dir,args.dataset,m,args.task_num,args.seed)
            print("load model:",model_dir)
            model=get_model(model_dir,dataset,args.seed,train1,args.nonlinear)
            acc=eval(val_data,model)
            result_list[d].append(acc.item())
    
    print(result_list)
    acc_file='%s/acc.txt' % (args.save_dir)
    print("save acc:",acc_file)
    with open(acc_file, 'w') as file:
        file.write(json.dumps(result_list))
