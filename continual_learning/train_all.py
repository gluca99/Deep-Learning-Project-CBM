import torch,torchvision
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
from avalanche.benchmarks.datasets import CIFAR10,CIFAR100,TinyImagenet,CORe50Dataset,CUB200
from avalanche.benchmarks.classic import SplitFMNIST, SplitImageNet
from avalanche.benchmarks.classic.clear import CLEAR
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin

from avalanche.benchmarks import SplitMNIST,PermutedMNIST
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin,GEMPlugin, EWCPlugin, LwFPlugin, MIRPlugin
from avalanche.training.supervised import Naive, LwF, EWC,JointTraining,SynapticIntelligence,GenerativeReplay,ICaRL,GEM,AGEM,GDumb,MIR #,DER
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net, SimpleMLP
from collections import OrderedDict

import sys,os
sys.path.append(sys.path[0] + '/..')
from experiment import oldalexnet, oldvgg16, oldresnet152
import new_plugins,CIN_plugins
import random
import numpy as np
import argparse
import copy
import json

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--model', choices=['alexnet', 'vgg16','resnet152','resnet18','resnet34'],
            default='alexnet')
    aa('--pretrain',type=str,default='none')
    aa('--dataset', choices=['imagenet10','cifar10','broden','cifar100','SplitMNIST','PermutedMNIST','CLEAR','TinyImagenet','CORe50Dataset','CUB200','SplitFMNIST'],
            default='cifar100')
    aa('--ck_dir',default='./ck_dir/')
    aa('--cache',default='./cache/')
    aa('--download_data',default='./download_data')
    aa('--dataset_dir',default='./dataset_dir/')
    aa('--probing_method',type=str,choices=['NetDissect','CLIP-Dissect','Broden'],default='NetDissect')
    aa('--result_dir',default='./result_set/alexnet_cifar100_5task_20epoch')
    aa('--task_num',type=int,default=5)
    aa('--batch_size',type=int,default=256)
    aa('--epoch',type=int,default=20)
    aa('--device',default="cuda")
    aa('--lr',type=float,default=0.0005,help='learning rate')
    aa('--strategy',choices=['DER','SRT_DER','CBM_DER','EWC','GenerativeReplay','SI','LwF', 'ICaRL','GEM', 'AGEM', 'Naive','JointTraining','EWC_based','CIN','EWC_based_GEM','EWC_based_CIN','SRT','SRT_GEM','CBM_GEM','GDumb','SRT_GDumb','CBM_GDumb','MIR','SRT_MIR','CBM_MIR','LwF_GEM','LwF_MIR','EWC_GEM','EWC_MIR'],default='EWC_based')
    aa("--ewc_mode", type=str, choices=["separate", "online"], default="online", help="Choose between EWC and online.")
    aa("--ewc_lambda",type=float, default=0.4,help="Penalty hyperparameter for EWC",)
    aa("--sparse_lam",type=float, default=0.00007,help="Penalty hyperparameter for sparse final layer",)
    aa("--gem_mem",type=int,default=150,help="gem buffer size per exp")
    aa("--decay_factor",type=float,default=0.1,help="Decay factor for importance " "when ewc_mode is online.",)
    aa("--l1_coeff",type=float,default=1,help="l1 penalty coefficient for SRT strategy")
    aa("--zero_threshold",type=float,default=0.01,help="neurons connectivity threshold for SRT strategy")
    aa("--alpha",type=float,default=1,help="b_i for p before concept unit")
    aa("--beta",type=float,default=0.8,help="b_i for p for concept unit")
    aa("--gamma",type=float,default=1,help="b_i for p for non-concept unit and fc layers")
    aa("--random",type=bool,default=False,help='freeze random unit in SRT method')
    aa("--sol",type=str,default="sol0",help='implementation methods for freezing subnetwork')
    aa("--dif",type=str,default="False",help='when trying different buffer size for CBM_GEM, use the original W_c, W_f instead')
    aa("--nonlinear",type=str,default="False",help='Nonlinear W_c')
    aa("--seed",type=str,default="1234",help="seed for spliting class")
    aa("--train1",type=str,default="False",help="train for only one task")
    aa("--sparse",type=str,default="False",help="Sparse final layer")
    aa('--timestamp',type=int,default=20)
    aa('--evaluation_protocol',default='streaming',help="CLEAR dataset mode, iid or streaming")
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    #if args.strategy== 'ICaRL':
    #    torch.backends.cudnn.enabled = False
    # result directory
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.ck_dir):
        os.makedirs(args.ck_dir)

    args_dict_tmp = vars(args)
    args_dict = copy.deepcopy(args_dict_tmp)
    with open(os.path.join(args.result_dir, "param.txt"), mode="w") as f:
        f.write("============ parameters ============\n")
        print("============ parameters =============")
        for k, v in args_dict.items():
            f.write("{}: {}\n".format(k, v))
            print("{}: {}".format(k, v))
    # device
    print(f"************* {args.strategy} *************")
    device = torch.device(args.device  if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    if args.dataset in ['imagenet10','cifar10','cifar100','CLEAR','TinyImagenet','CORe50Dataset','CUB200','SplitMNIST']:
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
        if args.dataset=='cifar10':
            train_data=CIFAR10(args.download_data,train=True,download=True,transform=train_transforms)
            test_data=CIFAR10(args.download_data,train=False,download=True,transform=test_transforms)
            scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, shuffle=True, seed=args.seed,task_labels=False)
        elif args.dataset == 'cifar100':
            train_data=CIFAR100(args.download_data,train=True,download=True,transform=train_transforms)
            test_data=CIFAR100(args.download_data,train=False,download=True,transform=test_transforms)
            if 'fix' not in args.seed:
                scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, shuffle=True, seed=args.seed,task_labels=False)
            else:
                order_list=[]
                if seed=='fix':
                    order_file=CACHE+'%s_order.txt'%('cifar100')
                else:
                    order_file=CACHE+'%s_order_%s.txt'%('cifar100',seed)
                with open(order_file, 'r') as file:
                    for line in file:
                        order_list.append(int(line))
                scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, fixed_class_order=order_list,task_labels=False)
        elif args.dataset == 'TinyImagenet':
            train_data=TinyImagenet(args.download_data,train=True,download=True,transform=train_transforms)
            test_data=TinyImagenet(args.download_data,train=False,download=True,transform=test_transforms)
            if 'fix' not in args.seed:
                scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, shuffle=True, seed=args.seed,task_labels=False)
            else:
                order_list=[]
                if seed=='fix':
                    order_file=CACHE+'%s_order.txt'%('cifar100')
                else:
                    order_file=CACHE+'%s_order_%s.txt'%('cifar100',seed)
                with open(order_file, 'r') as file:
                    for line in file:
                        order_list.append(int(line))
                scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, fixed_class_order=order_list,task_labels=False)
        elif args.dataset == 'CORe50Dataset':
            train_data=CORe50Dataset(args.download_data,train=True,download=True,transform=train_transforms)
            test_data=CORe50Dataset(args.download_data,train=False,download=True,transform=test_transforms)
            scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, shuffle=True, seed=args.seed,task_labels=False)
        elif args.dataset == 'CUB200':
            train_data=CUB200(args.download_data,train=True,download=True,transform=train_transforms)
            test_data=CUB200(args.download_data,train=False,download=True,transform=test_transforms)
            scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num, shuffle=True, seed=args.seed,task_labels=False)
        elif args.dataset == 'imagenet10':
            train_data=ImageFolder('./download_data/imagenet/imagenet10/train',transform=train_transforms)
            test_data=ImageFolder('./download_data/imagenet/imagenet10/val',transform=test_transforms)
            print("************ class to index ***********")
            print(train_data.class_to_idx)
            print(test_data.class_to_idx)
            scenario = nc_benchmark(train_data, test_data, n_experiences=args.task_num,task_labels=False)
        elif args.dataset == 'SplitMNIST':
            train_transforms=transforms.Compose([
                transforms.Resize((resize,resize)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
                transforms.Normalize(mean=means,
                                        std=stds)
            ])
            scenario = SplitMNIST(n_experiences=args.task_num,train_transform=train_transforms,eval_transform=train_transforms, seed=args.seed)
        else:
            scenario = CLEAR( data_name='clear10', evaluation_protocol=args.evaluation_protocol,feature_type=None,
                                train_transform=train_transforms,eval_transform=test_transforms,dataset_root=args.download_data)
    elif args.dataset == 'PermutedMNIST':
        scenario = PermutedMNIST(n_experiences=args.task_num, seed=args.seed)
    else:
        scenario= SplitFMNIST(n_experiences=args.task_num, seed=args.seed,dataset_root=args.download_data)
    # MODEL CREATION
    n_classes=11 if args.dataset=='CLEAR' else scenario.n_classes
    n_classes=365 if args.pretrain!='none' else scenario.n_classes
    #n_classes=1000 if args.pretrain=='img' else scenario.n_classes
    model_factory = dict(
        alexnet=oldalexnet.AlexNet,
        vgg16=oldvgg16.vgg16,
        resnet152=oldresnet152.OldResNet152,
        resnet18=torchvision.models.__dict__['resnet18'](pretrained=False,num_classes=n_classes),
        resnet34=torchvision.models.__dict__['resnet34'](pretrained=False,num_classes=n_classes))[args.model]
    if args.model not in ['resnet18','resnet34']:
        model=model_factory(num_classes=n_classes,split_groups=False)
    else:
        if(args.strategy not in ['CBM_GEM','CBM_GDumb','CBM_MIR','CBM_DER']):
            model=model_factory
            if args.pretrain!='none':
                if args.pretrain!= 'img':
                    state_dict = torch.load('./sandbox-lf-cbm/data/resnet18_places365.pth.tar')['state_dict']
                    new_state_dict = {}
                    for key in state_dict:
                        if key.startswith('module.'):
                            new_state_dict[key[7:]] = state_dict[key]
                    model.load_state_dict(new_state_dict)
                    n_classes=scenario.n_classes
                    model.fc=nn.Linear(512, n_classes)
                else:
                    if args.model=='resnet18':
                        model=torchvision.models.__dict__['resnet18'](pretrained=True,num_classes=n_classes)
                    else:
                        model=torchvision.models.__dict__['resnet34'](pretrained=True,num_classes=n_classes)
                    n_classes=scenario.n_classes
                    model.fc=nn.Linear(512, n_classes)

        else:
            back=model_factory
            if args.pretrain != 'img':
                state_dict = torch.load('./sandbox-lf-cbm/data/resnet18_places365.pth.tar')['state_dict']
                new_state_dict = {}
                for key in state_dict:
                    if key.startswith('module.'):
                        new_state_dict[key[7:]] = state_dict[key]
                back.load_state_dict(new_state_dict)
            else:
                if args.model=='resnet18':
                    back=torchvision.models.__dict__['resnet18'](pretrained=True)
                else:
                    back=torchvision.models.__dict__['resnet34'](pretrained=True)
            freeze=False
            if args.strategy=='CBM_GEM':
                naive_path=args.result_dir.replace("CBM_GEM","cc_cbm")
            elif args.strategy=='CBM_DER':
                naive_path=args.result_dir.replace("CBM_DER","cc_cbm")
            elif args.strategy=='CBM_GDumb':
                naive_path=args.result_dir.replace("freeze_reg_gdumb","freeze_reg")
            elif args.strategy=='CBM_MIR':
                freeze=False
                naive_path=args.result_dir.replace("CBM_MIR","cc_cbm")

            if args.dif!='False':
                naive_path=naive_path.replace("/dif","")
            w_f_path='%s/%s_task_%d_%d_%s_cbm/W_g.pt' % (naive_path,args.dataset,args.task_num-1,args.task_num,args.seed)
            W_f=torch.load(w_f_path).to(args.device)

            w_c_path='%s/%s_task_%d_%d_%s_cbm/W_c.pt' % (naive_path,args.dataset,args.task_num-1,args.task_num,args.seed)
            W_c=torch.load(w_c_path).to(args.device)

            mean_path='%s/%s_task_%d_%d_%s_cbm/proj_mean.pt' % (naive_path,args.dataset,args.task_num-1,args.task_num,args.seed)
            mean=torch.load(mean_path).to(args.device)
            std_path='%s/%s_task_%d_%d_%s_cbm/proj_std.pt' % (naive_path,args.dataset,args.task_num-1,args.task_num,args.seed)
            std=torch.load(std_path).to(args.device)

            model=CBM_model(back,W_c,mean,std,W_f,args.nonlinear,freeze)
    print(f'The model has {count_parameters(model):,} trainable parameters')


    # log to Tensorboard
    tb_logger = TensorboardLogger()

    # log to text file
    log_file="%s/%s_log.txt" %(args.result_dir,args.strategy)
    text_logger = TextLogger(open(log_file, 'w'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=n_classes, save_image=False,
                                stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )

    optimizer=optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.1
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    if args.strategy=='EWC':
        cl_strategy = EWC(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(), args.ewc_lambda, args.ewc_mode, decay_factor=args.decay_factor,
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size,plugins=[new_plugins.SPARSE_Plugin()] if args.sparse=='True' else None, evaluator=eval_plugin)
    elif args.strategy=='GenerativeReplay':
        cl_strategy = GenerativeReplay(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='SI':
        cl_strategy = SynapticIntelligence(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(), si_lambda=args.ewc_lambda,
            train_mb_size=args.batch_size, train_epochs=args.epoch, eval_mb_size=args.batch_size,evaluator=eval_plugin,device=device)
    elif args.strategy=='LwF':
        cl_strategy = LwF(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(), alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,train_mb_size=args.batch_size, 
            train_epochs=args.epoch, eval_mb_size=args.batch_size,evaluator=eval_plugin,device=device)
    elif args.strategy=='ICaRL':
        # copy from https://github.com/ContinualAI/avalanche/blob/master/examples/icarl.py
        # TODO: need to understand the mechanism of this strategy
        model: IcarlNet = make_icarl_net(num_classes=10)
        model.apply(initialize_icarl_net)
        cl_strategy = ICaRL(
            model.feature_extractor,model.classifier, optim.Adam(model.parameters(), lr=args.lr),
            memory_size=2000,buffer_transform=transforms.Compose([icarl_cifar100_augment_data]),fixed_memory=True,
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, eval_mb_size=1, evaluator=eval_plugin)
    elif args.strategy=='GEM':
        buffer_size=args.gem_mem
        cl_strategy = GEM(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),patterns_per_exp=args.gem_mem,memory_strength=0.5,
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size,plugins=[new_plugins.SPARSE_Plugin()] if args.sparse=='True' else None, evaluator=eval_plugin)
    elif args.strategy=='AGEM':
        buffer_size=args.gem_mem
        cl_strategy = AGEM(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),patterns_per_exp=buffer_size,sample_size=buffer_size,
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='GDumb':
        cl_strategy = GDumb(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),mem_size=args.gem_mem, 
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='MIR':
        cl_strategy = MIR(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),mem_size=args.gem_mem,subsample=args.gem_mem,
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='DER':
        cl_strategy = DER(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),mem_size=args.gem_mem,
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='Naive':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size,plugins=[new_plugins.SPARSE_Plugin()] if args.sparse=='True' else None, evaluator=eval_plugin)
    elif args.strategy=='EWC_based':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[new_plugins.EWC_BASE_Plugin(args.ewc_lambda,args.model,args.dataset,args.alpha,
            args.beta,args.gamma,'online',args.decay_factor,task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir,probing_broden=args.probing_broden)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='CIN':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[CIN_plugins.CIN_Plugin(args.model,args.dataset,
            task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir,probing_broden=args.probing_broden)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='SRT':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[new_plugins.SRT_Plugin(args.model,args.dataset,args.l1_coeff,args.zero_threshold,
            task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir,probing_method=args.probing_method,random=args.random,strategy=args.strategy,sol=args.sol,n_classes=n_classes),new_plugins.EWC_BASE_Plugin(args.ewc_lambda,args.model,args.dataset,args.alpha,args.beta,args.gamma,'online',args.decay_factor,task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='SRT_GEM':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[new_plugins.SRT_Plugin(args.model,args.dataset,args.l1_coeff,args.zero_threshold,
            task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir,probing_method=args.probing_method,random=args.random,strategy=args.strategy,sol=args.sol,n_classes=n_classes),
            new_plugins.EWC_BASE_Plugin(args.ewc_lambda,args.model,args.dataset,args.alpha,args.beta,args.gamma,'online',args.decay_factor,task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir),
            GEMPlugin(patterns_per_experience=args.gem_mem,memory_strength=0.5)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='EWC_GEM':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[EWCPlugin(args.ewc_lambda, args.ewc_mode, decay_factor=args.decay_factor),
            GEMPlugin(patterns_per_experience=args.gem_mem,memory_strength=0.5)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='LwF_GEM':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[LwFPlugin(alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1),
            GEMPlugin(patterns_per_experience=args.gem_mem,memory_strength=0.5)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='CBM_GEM':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[new_plugins.CBM_Plugin(args.ewc_lambda,args.sparse_lam,args.model,args.dataset,args.seed,args.gamma,args.dif,args.nonlinear,args.task_num,'gem','online',args.decay_factor,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir),
            GEMPlugin(patterns_per_experience=args.gem_mem,memory_strength=0.5)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='SRT_MIR': 
        cl_strategy = MIR(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),mem_size=args.gem_mem,subsample=args.gem_mem,plugins=[new_plugins.SRT_Plugin(args.model,args.dataset,args.l1_coeff,args.zero_threshold,
            task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir,probing_method=args.probing_method,random=args.random,strategy=args.strategy,sol=args.sol,n_classes=n_classes),
            new_plugins.EWC_BASE_Plugin(args.ewc_lambda,args.model,args.dataset,args.alpha,args.beta,args.gamma,'online',args.decay_factor,task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='EWC_MIR':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[EWCPlugin(args.ewc_lambda, args.ewc_mode, decay_factor=args.decay_factor),
            MIRPlugin(mem_size=args.gem_mem,subsample=args.gem_mem)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='LwF_MIR':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[LwFPlugin(alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1),
            MIRPlugin(mem_size=args.gem_mem,subsample=args.gem_mem)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='CBM_MIR': 
        cl_strategy = MIR(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),mem_size=args.gem_mem,subsample=args.gem_mem,plugins=[new_plugins.CBM_Plugin(args.ewc_lambda,args.sparse_lam,args.model,args.dataset,args.seed,args.gamma,args.dif,args.nonlinear,args.task_num,'mir','online',args.decay_factor,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='SRT_DER': 
        cl_strategy =DER(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),mem_size=args.gem_mem,plugins=[new_plugins.SRT_Plugin(args.model,args.dataset,args.l1_coeff,args.zero_threshold,
            task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir,probing_method=args.probing_method,random=args.random,strategy=args.strategy,sol=args.sol,n_classes=n_classes),
            new_plugins.EWC_BASE_Plugin(args.ewc_lambda,args.model,args.dataset,args.alpha,args.beta,args.gamma,'online',args.decay_factor,task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='CBM_DER': 
        cl_strategy = DER(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),mem_size=args.gem_mem,plugins=[new_plugins.CBM_Plugin(args.ewc_lambda,args.sparse_lam,args.model,args.dataset,args.seed,args.gamma,args.dif,args.nonlinear,args.task_num,replay='der',mode='online',decay_factor=args.decay_factor,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='EWC_based_GEM':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[new_plugins.EWC_BASE_Plugin(args.ewc_lambda,args.model,args.dataset,args.alpha,
            args.beta,args.gamma,'online',args.decay_factor,task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir),GEMPlugin(patterns_per_experience=150,memory_strength=0.5)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    elif args.strategy=='EWC_based_CIN':
        cl_strategy = Naive(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),plugins=[new_plugins.EWC_BASE_Plugin(args.ewc_lambda,args.model,args.dataset,args.alpha,
            args.beta,args.gamma,'online',args.decay_factor,task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir),
            CIN_plugins.CIN_Plugin(args.model,args.dataset,
            task_num=args.task_num,ck_dir=args.ck_dir,result_dir=args.result_dir,dataset_dir=args.dataset_dir)],
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)
    else:
        # JointTraining
        cl_strategy = JointTraining(
            model, optim.Adam(model.parameters(), lr=args.lr),
            CrossEntropyLoss(),
            train_epochs=args.epoch, device=device, train_mb_size=args.batch_size, evaluator=eval_plugin)

    # TRAINING LOOP
    print('Starting experiment...')
    results_list = {}
    for i in range(args.task_num):
      results_list[i]=[]
    if args.strategy == 'JointTraining':
      cl_strategy.train(scenario.train_stream)
      results=cl_strategy.eval(scenario.test_stream)
      for exp in range(args.task_num):
          key='Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00%d'%(exp)
          results_list[exp].append(results[key])
      print(results_list)
    else:
        for experience in scenario.train_stream:
            #save dataset to torch.Dataset

            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            if experience.current_experience>=1 and args.train1=='True':
                break
            with open(os.path.join(args.result_dir, "param.txt"), mode="a") as f:
                f.write(f"task {experience.current_experience}:\n")
                f.write(f"{experience.classes_in_this_experience}\n")

            # train returns a dictionary which contains all the metric values
            res = cl_strategy.train(experience)
            print('Training completed')
            #save model
            if args.strategy not in ['EWC_based','SRT','SRT_GEM']:
                model_save_path='%s%s-%s-%s-task%d-%d.pt' % (args.ck_dir,args.model,args.strategy,args.dataset,experience.current_experience,args.task_num)
                torch.save(model.state_dict(), model_save_path)

            print('Computing accuracy on the whole test set')
            # test also returns a dictionary which contains all the metric values
            results=cl_strategy.eval(scenario.test_stream)
            for exp in range(experience.current_experience+1):
                if exp<10:
                    key='Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00%d'%(exp)
                else:
                    key='Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp0%d'%(exp)
                if args.dataset in ['PermutedMNIST','CLEAR']:
                    key='Top1_Acc_Exp/eval_phase/test_stream/Task00%d/Exp00%d'%(exp,exp)
                results_list[exp].append(results[key])
            print(results_list)
    
            print("save test_acc")
            if (args.strategy not in ['SRT','SRT_GEM']):
                acc_file="%s/%s_acc.txt" %(args.result_dir,args.strategy)
            elif (args.random==False):
                acc_file="%s/%s_%s_acc.txt" %(args.result_dir,args.strategy,args.sol)
            else:
                acc_file="%s/%s_%s_random_acc.txt" %(args.result_dir,args.strategy,args.sol)
            with open(acc_file, 'w') as file:
                file.write(json.dumps(results_list)) 
    
    with open(os.path.join(args.result_dir, "param.txt"), mode="a") as f:
        f.write(f"Max GPU usage: {torch.cuda.max_memory_allocated('cuda')}:\n")
    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# copy from https://github.com/ContinualAI/avalanche/blob/master/examples/icarl.py
def icarl_cifar100_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ]
    else:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ][:, :, ::-1]
    t = torch.tensor(random_cropped)
    return t

class CBM_model(torch.nn.Module):
    def __init__(self, model,W_c, proj_mean, proj_std, W_g, nonlinear,freeze, device="cuda"):
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        self.backbone.requires_grad=True    
        if(nonlinear=='False'):
            self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
            self.proj_layer.weight=nn.parameter.Parameter(torch.zeros(self.proj_layer.weight.shape).to(device))
            if freeze ==True:
                self.proj_layer.weight.requires_grad=False
        else:
            self.proj_layer = torch.nn.Sequential(OrderedDict([
            ('first', torch.nn.Linear(in_features=W_c.first.weight.shape[1], out_features=4096,bias=True)),
            ('relu1',torch.nn.ReLU()),
            ('cbl', torch.nn.Linear(in_features=4096, out_features=W_c.cbl.weight.shape[0],bias=False))
            ])).to(device)
            self.proj_layer.first.weight=W_c.first.weight
            self.proj_layer.first.bias=W_c.first.bias
            self.proj_layer.cbl.weight=nn.parameter.Parameter(torch.zeros(self.proj_layer.cbl.weight.shape).to(device))
            if freeze ==True:
                self.proj_layer.first.weight.requires_grad=False 
                self.proj_layer.first.bias.requires_grad=False 
                self.proj_layer.cbl.weight.requires_grad=False 
        self.proj_mean = torch.zeros(proj_mean.shape).to(device)
        self.proj_std = torch.nn.init.ones_(proj_std).to(device)
        if freeze ==True:
            self.proj_mean.requires_grad=False
            self.proj_std.requires_grad=False
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.concepts = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x
    

if __name__ == '__main__':
    main()
    print("Max GPU usage:",torch.cuda.max_memory_allocated('cuda'))
    #with open(os.path.join(args.result_dir, "param.txt"), mode="a") as f:
    #    f.write(f"Max GPU usage: {torch.cuda.max_memory_allocated('cuda')}:\n")

    
