import os
import torch
import pandas as pd
import numpy as np
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from avalanche.benchmarks.datasets import CIFAR10,CIFAR100,TinyImagenet,CUB200
from avalanche.benchmarks.generators import nc_benchmark
#from modified_represent import resnet18_rep

import clip
from pytorchcv.model_provider import get_model as ptcv_get_model

DATASET_ROOTS = {
    "imagenet_train": os.path.expanduser("/ceph")+"/shared_read_only/imagenet/ILSVRC/Data/CLS-LOC/train/",
    "imagenet_val": os.path.expanduser("/ceph")+"/OOD_detection/imagenet-r/DeepAugment/ImageNet_val/",
    "places365_train": "data/places365/train",
    "places365_val": "data/places365/val",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test"
}

DOWNLOAD="./download_data"
CACHE="./cache/"

LABEL_FILES = {"places365":"data/categories_places365_clean.txt",
               "imagenet":"data/imagenet_classes.txt",
               "cifar10":"data/cifar10_classes.txt",
               "cifar100":"data/cifar100_classes.txt",
               "cifar100_task_0_3456":"./sandbox-lf-cbm/data/cifar100_task_0_3456_classes.txt",
               "cifar100_task_1_3456":"./sandbox-lf-cbm/data/cifar100_task_1_3456_classes.txt",
               "cifar100_task_2_3456":"./sandbox-lf-cbm/data/cifar100_task_2_3456_classes.txt",
               "cifar100_task_3_3456":"./sandbox-lf-cbm/data/cifar100_task_3_3456_classes.txt",
               "cifar100_task_4_3456":"./sandbox-lf-cbm/data/cifar100_task_4_3456_classes.txt",
               "cifar100_task_0_5678":"./sandbox-lf-cbm/data/cifar100_task_0_5678_classes.txt",
               "cifar100_task_1_5678":"./sandbox-lf-cbm/data/cifar100_task_1_5678_classes.txt",
               "cifar100_task_2_5678":"./sandbox-lf-cbm/data/cifar100_task_2_5678_classes.txt",
               "cifar100_task_3_5678":"./sandbox-lf-cbm/data/cifar100_task_3_5678_classes.txt",
               "cifar100_task_4_5678":"./sandbox-lf-cbm/data/cifar100_task_4_5678_classes.txt",
               "cifar100_task_0_fix":"./sandbox-lf-cbm/data/cifar100_task_0_fix_classes.txt",
               "cifar100_task_1_fix":"./sandbox-lf-cbm/data/cifar100_task_1_fix_classes.txt",
               "cifar100_task_2_fix":"./sandbox-lf-cbm/data/cifar100_task_2_fix_classes.txt",
               "cifar100_task_3_fix":"./sandbox-lf-cbm/data/cifar100_task_3_fix_classes.txt",
               "cifar100_task_4_fix":"./sandbox-lf-cbm/data/cifar100_task_4_fix_classes.txt",
               "cifar100_task_0_fix2":"./sandbox-lf-cbm/data/cifar100_task_0_fix2_classes.txt",
               "cifar100_task_1_fix2":"./sandbox-lf-cbm/data/cifar100_task_1_fix2_classes.txt",
               "cifar100_task_2_fix2":"./sandbox-lf-cbm/data/cifar100_task_2_fix2_classes.txt",
               "cifar100_task_3_fix2":"./sandbox-lf-cbm/data/cifar100_task_3_fix2_classes.txt",
               "cifar100_task_4_fix2":"./sandbox-lf-cbm/data/cifar100_task_4_fix2_classes.txt",
               "TinyImagenet_task_0_3456":"./sandbox-lf-cbm/data/TinyImagenet_task_0_3456_classes.txt",
               "TinyImagenet_task_1_3456":"./sandbox-lf-cbm/data/TinyImagenet_task_1_3456_classes.txt",
               "TinyImagenet_task_2_3456":"./sandbox-lf-cbm/data/TinyImagenet_task_2_3456_classes.txt",
               "TinyImagenet_task_3_3456":"./sandbox-lf-cbm/data/TinyImagenet_task_3_3456_classes.txt",
               "TinyImagenet_task_4_3456":"./sandbox-lf-cbm/data/TinyImagenet_task_4_3456_classes.txt",
               "TinyImagenet_task_0_5678":"./sandbox-lf-cbm/data/TinyImagenet_task_0_5678_classes.txt",
               "TinyImagenet_task_1_5678":"./sandbox-lf-cbm/data/TinyImagenet_task_1_5678_classes.txt",
               "TinyImagenet_task_2_5678":"./sandbox-lf-cbm/data/TinyImagenet_task_2_5678_classes.txt",
               "TinyImagenet_task_3_5678":"./sandbox-lf-cbm/data/TinyImagenet_task_3_5678_classes.txt",
               "TinyImagenet_task_4_5678":"./sandbox-lf-cbm/data/TinyImagenet_task_4_5678_classes.txt",
               "TinyImagenet_task_0_fix":"./sandbox-lf-cbm/data/TinyImagenet_task_0_fix_classes.txt",
               "TinyImagenet_task_1_fix":"./sandbox-lf-cbm/data/TinyImagenet_task_1_fix_classes.txt",
               "TinyImagenet_task_2_fix":"./sandbox-lf-cbm/data/TinyImagenet_task_2_fix_classes.txt",
               "TinyImagenet_task_3_fix":"./sandbox-lf-cbm/data/TinyImagenet_task_3_fix_classes.txt",
               "TinyImagenet_task_4_fix":"./sandbox-lf-cbm/data/TinyImagenet_task_4_fix_classes.txt",
               "cifar10_task_0_3456":"./sandbox-lf-cbm/data/cifar10_task_0_3456_classes.txt",
               "cifar10_task_1_3456":"./sandbox-lf-cbm/data/cifar10_task_1_3456_classes.txt",
               "cifar10_task_2_3456":"./sandbox-lf-cbm/data/cifar10_task_2_3456_classes.txt",
               "cifar10_task_3_3456":"./sandbox-lf-cbm/data/cifar10_task_3_3456_classes.txt",
               "cifar10_task_4_3456":"./sandbox-lf-cbm/data/cifar10_task_4_3456_classes.txt",
               "cifar10_task_0_5678":"./sandbox-lf-cbm/data/cifar10_task_0_5678_classes.txt",
               "cifar10_task_1_5678":"./sandbox-lf-cbm/data/cifar10_task_1_5678_classes.txt",
               "cifar10_task_2_5678":"./sandbox-lf-cbm/data/cifar10_task_2_5678_classes.txt",
               "cifar10_task_3_5678":"./sandbox-lf-cbm/data/cifar10_task_3_5678_classes.txt",
               "cifar10_task_4_5678":"./sandbox-lf-cbm/data/cifar10_task_4_5678_classes.txt",
               "CUB200_task_0_3456":"./sandbox-lf-cbm/data/CUB200_task_0_3456_classes.txt",
               "CUB200_task_1_3456":"./sandbox-lf-cbm/data/CUB200_task_1_3456_classes.txt",
               "CUB200_task_2_3456":"./sandbox-lf-cbm/data/CUB200_task_2_3456_classes.txt",
               "CUB200_task_3_3456":"./sandbox-lf-cbm/data/CUB200_task_3_3456_classes.txt",
               "CUB200_task_4_3456":"./sandbox-lf-cbm/data/CUB200_task_4_3456_classes.txt",
               "CUB200_task_0_5678":"./sandbox-lf-cbm/data/CUB200_task_0_5678_classes.txt",
               "CUB200_task_1_5678":"./sandbox-lf-cbm/data/CUB200_task_1_5678_classes.txt",
               "CUB200_task_2_5678":"./sandbox-lf-cbm/data/CUB200_task_2_5678_classes.txt",
               "CUB200_task_3_5678":"./sandbox-lf-cbm/data/CUB200_task_3_5678_classes.txt",
               "CUB200_task_4_5678":"./sandbox-lf-cbm/data/CUB200_task_4_5678_classes.txt",
               "cub":"data/cub_classes.txt"}

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name,seed, clip_preprocess=None, target_preprocess=None, batch=None):
    print("get_data:",dataset_name)
    seg=dataset_name.split("_")
    if seg[1]=="task":
        if seg[0]=="cifar100":
            clip_train=CIFAR100(DOWNLOAD,train=True,download=True,transform=clip_preprocess)
            clip_test=CIFAR100(DOWNLOAD,train=False,download=True,transform=clip_preprocess)
            target_train=CIFAR100(DOWNLOAD,train=True,download=True,transform=target_preprocess)
            target_test=CIFAR100(DOWNLOAD,train=False,download=True,transform=target_preprocess)
        elif seg[0]=="imagenet10":
            clip_train=ImageFolder('./download_data/imagenet/imagenet10/train',transform=clip_preprocess)
            clip_test=ImageFolder('./download_data/imagenet/imagenet10/val',transform=clip_preprocess)
            target_train=ImageFolder('./download_data/imagenet/imagenet10/train',transform=target_preprocess)
            target_test=ImageFolder('./download_data/imagenet/imagenet10/val',transform=target_preprocess)
        elif seg[0]=="cifar10":
            clip_train=CIFAR10(DOWNLOAD,train=True,download=True,transform=clip_preprocess)
            clip_test=CIFAR10(DOWNLOAD,train=False,download=True,transform=clip_preprocess)
            target_train=CIFAR10(DOWNLOAD,train=True,download=True,transform=target_preprocess)
            target_test=CIFAR10(DOWNLOAD,train=False,download=True,transform=target_preprocess)
        elif seg[0]=="CUB200":
            clip_train=CUB200(DOWNLOAD+'/cub200',train=True,download=True,transform=clip_preprocess)
            clip_test=CUB200(DOWNLOAD+'/cub200',train=False,download=True,transform=clip_preprocess)
            target_train=CUB200(DOWNLOAD+'/cub200',train=True,download=True,transform=target_preprocess)
            target_test=CUB200(DOWNLOAD+'/cub200',train=False,download=True,transform=target_preprocess)
        elif seg[0]=="TinyImagenet":
            clip_train=TinyImagenet(DOWNLOAD,train=True,download=True,transform=clip_preprocess)
            clip_test=TinyImagenet(DOWNLOAD,train=False,download=True,transform=clip_preprocess)
            target_train=TinyImagenet(DOWNLOAD,train=True,download=True,transform=target_preprocess)
            target_test=TinyImagenet(DOWNLOAD,train=False,download=True,transform=target_preprocess)

        if "fix" not in seed:
            seed=int(seed)
            print('SEED:',seed)
            clip_scenario = nc_benchmark(clip_train, clip_test, n_experiences=int(seg[3]), shuffle=True, seed=seed,task_labels=False)
            target_scenario = nc_benchmark(target_train, target_test, n_experiences=int(seg[3]), shuffle=True, seed=seed,task_labels=False)
            if seg[0]== "imagenet10":
                clip_scenario = nc_benchmark(clip_train, clip_test, n_experiences=int(seg[3]), task_labels=False)
                target_scenario = nc_benchmark(target_train, target_test, n_experiences=int(seg[3]), task_labels=False)

        else:
            order_list=[]
            if seed=='fix':
                order_file=CACHE+'%s_order.txt'%(seg[0])
            else:
                order_file=CACHE+'%s_order_%s.txt'%(seg[0],seed)
            with open(order_file, 'r') as file:
                for line in file:
                    order_list.append(int(line))
            clip_scenario = nc_benchmark(clip_train, clip_test, n_experiences=int(seg[3]), fixed_class_order=order_list,task_labels=False)
            target_scenario = nc_benchmark(target_train, target_test, n_experiences=int(seg[3]), fixed_class_order=order_list,task_labels=False)
        
        task=int(seg[2])
        if(seg[-1]=='train'):
            label=[item[1] for item in clip_scenario.train_stream[task].dataset]
            print("label len in transform:",len(label))
            label=torch.tensor(label)
            clip_data=[item[0] for item in clip_scenario.train_stream[task].dataset]
            target_data=[item[0] for item in target_scenario.train_stream[task].dataset]
        else:
            label=[item[1] for item in clip_scenario.test_stream[task].dataset]
            label=torch.tensor(label)
            clip_data=[item[0] for item in clip_scenario.test_stream[task].dataset]
            target_data=[item[0] for item in target_scenario.test_stream[task].dataset]
        
        del clip_scenario, target_scenario
        # turn to float16 precision
        clip_data = [clip_data[i].to(torch.float16) for i in range(len(clip_data))]
        target_data = [target_data[i].to(torch.float16) for i in range(len(target_data))]

        clip_data=torch.stack(clip_data,dim=0)
        target_data=torch.stack(target_data,dim=0)
        
        data_c=torch.utils.data.TensorDataset(clip_data,label)
        data_t=torch.utils.data.TensorDataset(target_data,label)
        
    else:
        if dataset_name == "cifar100_train":
            data_c = datasets.CIFAR100(root=os.path.expanduser("/ceph/.cache"), download=True, train=True,
                                    transform=clip_preprocess)
            data_t = datasets.CIFAR100(root=os.path.expanduser("/ceph/.cache"), download=True, train=True,
                                    transform=target_preprocess)

        elif dataset_name == "cifar100_val":
            data_c = datasets.CIFAR100(root=os.path.expanduser("/ceph/.cache"), download=True, train=False, 
                                    transform=clip_preprocess)
            data_t = datasets.CIFAR100(root=os.path.expanduser("/ceph/.cache"), download=True, train=False, 
                                    transform=target_preprocess)
            
        elif dataset_name == "cifar10_train":
            data_c = datasets.CIFAR10(root=os.path.expanduser("/ceph/.cache"), download=True, train=True,
                                    transform=clip_preprocess)
            data_t = datasets.CIFAR10(root=os.path.expanduser("/ceph/.cache"), download=True, train=True,
                                    transform=target_preprocess)
            
        elif dataset_name == "cifar10_val":
            data_c = datasets.CIFAR10(root=os.path.expanduser("/ceph/.cache"), download=True, train=False,
                                    transform=clip_preprocess)
            data_t = datasets.CIFAR10(root=os.path.expanduser("/ceph/.cache"), download=True, train=False,
                                    transform=target_preprocess)
            
        elif dataset_name in DATASET_ROOTS.keys():
            root = DATASET_ROOTS[dataset_name]
            data_c = datasets.ImageFolder(root, clip_preprocess)
            data_t = datasets.ImageFolder(root, target_preprocess)
                
        elif dataset_name == "imagenet_broden":
            root_i = DATASET_ROOTS["imagenet_val"]
            root_b = DATASET_ROOTS["broden"]
            data_c = torch.utils.data.ConcatDataset([datasets.ImageFolder(root_i, clip_preprocess), 
                                                        datasets.ImageFolder(root_b, clip_preprocess)])
            data_t = torch.utils.data.ConcatDataset([datasets.ImageFolder(root_i, target_preprocess), 
                                                        datasets.ImageFolder(root_b, target_preprocess)])
        
    return data_c, data_t

def get_pil_data(dataset_name,seed):
    seg=dataset_name.split("_")
    if seg[1]=="task":
        if seg[0]=="cifar100":
            train=CIFAR100(DOWNLOAD,train=True,download=True)
            test=CIFAR100(DOWNLOAD,train=False,download=True)
        elif seg[0]=="imagenet10":
            train=ImageFolder('./download_data/imagenet/imagenet10/train')
            test=ImageFolder('./download_data/imagenet/imagenet10/val')
        elif seg[0]=="cifar10":
            train=CIFAR10(DOWNLOAD,train=True,download=True)
            test=CIFAR10(DOWNLOAD,train=False,download=True)
        elif seg[0]=="CUB200":
            train=CUB200(DOWNLOAD+'/cub200',train=True,download=True)
            test=CUB200(DOWNLOAD+'/cub200',train=False,download=True)
        elif seg[0]=="TinyImagenet":
            train=TinyImagenet(DOWNLOAD,train=True,download=True)
            test=TinyImagenet(DOWNLOAD,train=False,download=True)

        if "fix" not in seed:
            seed=int(seed)
            scenario = nc_benchmark(train, test, n_experiences=int(seg[3]), shuffle=True, seed=seed,task_labels=False)
            if seg[0]== "imagenet10":
                scenario = nc_benchmark(train, test, n_experiences=int(seg[3]), task_labels=False)
        else:
            order_list=[]
            if seed=='fix':
                order_file=CACHE+'%s_order.txt'%(seg[0])
            else:
                order_file=CACHE+'%s_order_%s.txt'%(seg[0],seed)
            with open(order_file, 'r') as file:
                for line in file:
                    order_list.append(int(line))
            scenario = nc_benchmark(train, test, n_experiences=int(seg[3]), fixed_class_order=order_list,task_labels=False)
        
        task=int(seg[2])
        if(seg[-1]=='train'):
            label=[item[1] for item in scenario.train_stream[task].dataset]
            print("get pil data shape:",len(label))
            #label=torch.tensor(label)
            data=[item[0] for item in scenario.train_stream[task].dataset]
        else:
            label=[item[1] for item in scenario.test_stream[task].dataset]
            #label=torch.tensor(label)
            data=[item[0] for item in scenario.test_stream[task].dataset]
        #tensor=torch.stack(data,dim=0)
        #data=np.stack(data)
        return data,label
    
    else:
        if dataset_name == "cifar100_train":
            pil_data= datasets.CIFAR100(root=os.path.expanduser("/ceph/.cache"), download=True, train=True)

        elif dataset_name == "cifar100_val":
            pil_data= datasets.CIFAR100(root=os.path.expanduser("/ceph/.cache"), download=True, train=False)
        
        elif dataset_name == "cifar10_train":
            pil_data= datasets.CIFAR10(root=os.path.expanduser("/ceph/.cache"), download=True, train=True)
            
        elif dataset_name == "cifar10_val":
            pil_data= datasets.CIFAR10(root=os.path.expanduser("/ceph/.cache"), download=True, train=False)

        elif dataset_name in DATASET_ROOTS.keys():
            root = DATASET_ROOTS[dataset_name]
            pil_data= datasets.ImageFolder(root)
            
        elif dataset_name == "imagenet_broden":
            root_i = DATASET_ROOTS["imagenet_val"]
            root_b = DATASET_ROOTS["broden"]
            pil_data= torch.utils.data.ConcatDataset([datasets.ImageFolder(root_i), datasets.ImageFolder(root_b)])
    return pil_data,None


def get_targets_only(dataset_name,seed):
    pil_data , label= get_pil_data(dataset_name,seed)
    if label !=None:
        return label
    else:
        return pil_data.targets

def get_target_model(target_name,d_probe,seed,train1,device,pretrain=False):
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_img':
        target_model = models.resnet18(pretrained=pretrain).to(device)
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name == 'resnet34':
        target_model = models.resnet34(pretrained=pretrain).to(device)
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name == 'resnet50':
        target_model = models.resnet50(pretrained=pretrain).to(device)
        preprocess = get_resnet_imagenet_preprocess()
    
    #elif target_name == 'resnet18_ssre':
    #    target_model = resnet18_rep(True).to(device)
    #    preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name == 'resnet18_places': 
        if train1=='False':
            target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
            if pretrain:
                state_dict = torch.load('./sandbox-lf-cbm/data/resnet18_places365.pth.tar')['state_dict']
                new_state_dict = {}
                for key in state_dict:
                    if key.startswith('module.'):
                        new_state_dict[key[7:]] = state_dict[key]
                target_model.load_state_dict(new_state_dict)
        else:
            Label={}
            Label['cifar10']=10
            Label['imagenet10']=10
            Label['cifar100']=100
            Label['CUB200']=200
            Label['TinyImagenet']=200
            seg=d_probe.split("_")
            target_model = models.resnet18(pretrained=pretrain, num_classes=Label[seg[0]]).to(device)
            ck_dir='./ck_dir/'
            ptname='SEED_%s/resnet18-Naive-%s-task0-5.pt' % (seed,seg[0])
            ptname=ck_dir+ptname
            target_model.load_state_dict(torch.load(ptname))
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess
