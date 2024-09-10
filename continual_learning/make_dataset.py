import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import torch.optim as optim
from avalanche.benchmarks.datasets import CIFAR10, CIFAR100,TinyImagenet 
from avalanche.benchmarks.classic import SplitFMNIST
from avalanche.benchmarks.classic.clear import CLEAR

from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, LwF, EWC,JointTraining,SynapticIntelligence,GenerativeReplay,ICaRL
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net, SimpleMLP

import sys,os
sys.path.append(sys.path[0] + '/..')
from experiment import oldalexnet, oldvgg16, oldresnet152

import random
import numpy as np
import argparse
import pickle
import json

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--model', choices=['alexnet', 'vgg16','resnet152','resnet18'],
            default='alexnet')
    aa('--dataset', choices=['CLEAR','cifar10','cifar100','broden','TinyImagenet','SplitFMNIST'],
            default='SplitFMNIST')
    aa('--ck_path',default='~/concept-driven-continual-learning/ck_dir/')
    aa('--download_data',default='~/concept-driven-continual-learning/download_data')
    aa('--dataset_dir',default='~/concept-driven-continual-learning/dataset_dir/')
    aa('--result_dir',default='~/concept-driven-continual-learning/results/')
    aa('--task_num',type=int,default=5)
    aa('--seed',type=int,default=3456)
    aa('--batch_size',default=64)
    aa('--epoch',default=40)
    aa('--lr',default=0.001,help='learning rate')
    aa('--strategy',choices=['EWC','GenerativeReplay','SynapticIntelligence','LwF', 'ICaRL', 'Naive','JointTraining'],default='EWC')
    aa(
        "--ewc_mode",
        type=str,
        choices=["separate", "online"],
        default="online",
        help="Choose between EWC and online.",
    )
    aa(
        "--ewc_lambda",
        type=float,
        default=0.4,
        help="Penalty hyperparameter for EWC",
    )
    aa(
        "--decay_factor",
        type=float,
        default=0.1,
        help="Decay factor for importance " "when ewc_mode is online.",
    )
    aa('--timestamp',type=int,default=10)
    aa('--evaluation_protocol',default='streaming',help="CLEAR dataset mode, iid or streaming")
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    crop_size= 227 if args.model=='alexnet' else 224
    resize= 256
    means=[0.485, 0.456, 0.406]
    stds= [0.229, 0.224, 0.225]
    train_transforms=transforms.Compose([
                    transforms.Resize((resize,resize)),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=means,
                                            std=stds)
                ])
    test_transforms = transforms.Compose([
                    transforms.Resize((resize,resize)),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=means,
                                            std=stds)
                ])
    if args.dataset=='cifar100':
        train_data=CIFAR100(args.download_data,train=True,download=True,transform=train_transforms)
        test_data=CIFAR100(args.download_data,train=False,download=True,transform=test_transforms)
        scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, shuffle=True, seed=SEED,task_labels=False)
    elif args.dataset == 'cifar10':
        train_data=CIFAR10(args.download_data,train=True,download=True,transform=train_transforms)
        test_data=CIFAR10(args.download_data,train=False,download=True,transform=test_transforms)
        scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, shuffle=True, seed=SEED,task_labels=False)
    elif args.dataset == 'TinyImagenet':
            train_data=TinyImagenet(args.download_data,train=True,download=True,transform=train_transforms)
            test_data=TinyImagenet(args.download_data,train=False,download=True,transform=test_transforms)
            scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, shuffle=True, seed=SEED,task_labels=False)
    elif args.dataset == 'CLEAR':
        scenario = CLEAR( data_name='clear10', evaluation_protocol=args.evaluation_protocol,feature_type=None,
                                train_transform=train_transforms,eval_transform=test_transforms,dataset_root=args.download_data)
    else:
        scenario= SplitFMNIST(n_experiences=args.task_num, seed=args.seed)

    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("save dataset")

        dataset_path='%s/%s_1_task%d_%d.pt' % (args.dataset_dir,args.dataset,experience.current_experience,args.task_num)
        lst2 = [item[0] for item in experience.dataset]
        lstl = [item[1] for item in experience.dataset] #for fmnist
        data=torch.stack(lst2,dim=0)
        #data=data.repeat(1,3,1,1) #for fmnist
        label=torch.tensor(lstl) #for fmnist
        print(type(data),data.shape)
        dataset=torch.utils.data.TensorDataset(data,label) #for fmnist
        torch.save(dataset,dataset_path)
            
    

if __name__ == '__main__':
    main()

    
