import torch
import os
import random
import utils
import data_utils
import similarity
import argparse
import datetime
import json
import copy

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from normal_train import Trainer
parser = argparse.ArgumentParser(description='Settings for creating CBM')


parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--seed", type=str, default="3456")
parser.add_argument("--train1", type=str, default="False",help="load backbone which only train on the first task")
parser.add_argument("--nonlinear", type=str, default="False",help="nonlinear W_c")
parser.add_argument("--concept_set", type=str, default=None, help="path to concept set name")
parser.add_argument("--concept_method", type=str, default="c1", help="updating concept set method")
parser.add_argument("--freeze_wc",type=str,default="False")
parser.add_argument("--normalize_wf",type=str,default="False")
parser.add_argument("--normalize_wf_method",type=str,default="all")
parser.add_argument("--backbone", type=str, default="resnet18_places", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")
parser.add_argument("--pretrain", action='store_true', help="flag to use pretrained backbone")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--solver", type=str, default="glm", help="use glm or gradient to train the prediction layer")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--gamma", type=float, default=0.2, help="Regularization parameter to prevent from change, higher->less change")
parser.add_argument("--num_epoch", type=int, default=40,help='epoch to train the final layer')
parser.add_argument("--lr", type=float, default=0.0005,help='learning rate to train the final layer')
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")
parser.add_argument("--kill_dup", type=str, default="False")

def remove_duplicates(input_list):
    # Initialize an empty set to keep track of seen elements
    seen = set()
    # Initialize an empty list to store the result with unique elements
    unique_list = []

    # Iterate through each element in the input list
    for element in input_list:
        # If the element has not been seen before, add it to the set and the unique list
        if element not in seen:
            seen.add(element)
            unique_list.append(element)

    return unique_list

def rearrange_list(A, B):
    # Create a set from B for faster lookup
    B_set = set(B)
    # Initialize an empty list to store the result
    result = []

    # Append all elements of B to the result list
    result.extend(B)
    
    # Append the remaining elements of A that are not in B
    result.extend([element for element in A if element not in B_set])

    return result

def train_cbm_and_save(args):
    args_dict_tmp = vars(args)
    args_dict = copy.deepcopy(args_dict_tmp)
    print("============ parameters =============")
    for k, v in args_dict.items():
        print("{}: {}".format(k, v))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.concept_set==None:
        args.concept_set = "data/concept_sets/{}_gpt3_ensemble2.txt".format(args.dataset)
    similarity_fn = similarity.cos_similarity_cubed_single
    
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    
    #get concept set
    #cls_file="./sandbox-lf-cbm/data/%s_classes.txt" % (args.dataset)
    #cls_file = data_utils.LABEL_FILES[args.dataset]
    # print("Classes:",cls_file)
    # with open(cls_file, "r") as f:
    #     classes = f.read().split("\n")
    
    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    seg=args.dataset.split("_")
    #CBM Continual
    task=int(seg[2])

    print(f"Number of raw concepts for task {task}: {len(concepts)}")

    if args.kill_dup=="True":
        concepts=remove_duplicates(concepts)
        if task>0:
            previous_dataset=args.dataset.replace("task_%d"%(task),"task_%d"%(task-1))
            inter_concept_path="{}/{}_cbm/concepts.txt".format(args.save_dir,previous_dataset)
            print("load concept.txt: ",inter_concept_path)
            with open(inter_concept_path) as f:
                inter_concept_pre=f.read().split("\n")
            concepts=rearrange_list(concepts,inter_concept_pre)

    local_activation_dir=args.save_dir+"/"+args.activation_dir
    #save activations and get save_paths
    for d_probe in [d_train, d_val]:
        utils.save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe, seed=args.seed, train1=args.train1,
                               concept_set = args.concept_set, batch_size = args.batch_size, pretrain=args.pretrain,
                               device = args.device, pool_mode = "avg", save_dir = local_activation_dir,words=concepts)

    target_save_name, clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,d_train, args.concept_set, "avg", local_activation_dir)

    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", local_activation_dir)

    #load features
    with torch.no_grad():
        # only load target features from frozen backbone if running with pretrained model
        target_features = torch.load(target_save_name, map_location="cpu").float()
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()

        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features
    
    #filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)
    
    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i]<=args.clip_cutoff:
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.clip_cutoff]
    
    print(f"Number of concepts for task {task} after CLIP top 5: {len(concepts)}")

    #save memory by recalculating
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[highest>args.clip_cutoff]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
    
        clip_features = image_features @ text_features.T
        del image_features, text_features
    
    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]
    
    #learn projection layer
    if (args.nonlinear=='False'):
        proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                    bias=False).to(args.device)
    else:
        proj_layer = torch.nn.Sequential(OrderedDict([
            ('first', torch.nn.Linear(in_features=target_features.shape[1], out_features=4096,bias=True)),
            ('relu1',torch.nn.ReLU()),
            ('cbl', torch.nn.Linear(in_features=4096, out_features=len(concepts),bias=False))
            ])).to(args.device)
    
    if(task>0):
        with torch.no_grad():
            #Load previous W_c 
            previous_dataset=args.dataset.replace("task_%d"%(task),"task_%d"%(task-1))
            w_c_path="{}/{}_cbm/W_c.pt".format(args.save_dir,previous_dataset)
            print("load Wc: ",w_c_path)
            W_c_pre=torch.load(w_c_path).to(args.device)
            if (args.nonlinear!='False'):
                proj_layer.first.weight=W_c_pre.first.weight
                proj_layer.first.bias=W_c_pre.first.bias
            inter_concept_path="{}/{}_cbm/concepts.txt".format(args.save_dir,previous_dataset)
            print("load concept.txt: ",inter_concept_path)
            with open(inter_concept_path) as f:
                inter_concept_pre=f.read().split("\n")
            # concept set method 1
            if args.concept_method =="c1":
                concept_memory={}
                for i,con in enumerate(concepts):
                    seen=0
                    if(con in concept_memory.keys()):
                        seen=concept_memory[con]
                    for j,inter in enumerate(inter_concept_pre): # for the section of task t
                        if(con==inter):
                            if(seen==0):
                                if(con in concept_memory.keys()):
                                    concept_memory[con]+=1
                                else:
                                    concept_memory[con]=1
                                #print("assign previous concept:",i)
                                if (args.nonlinear=='False'):
                                    proj_layer.weight[i,:]=W_c_pre[j,:]
                                else:
                                    proj_layer.cbl.weight[i,:]=W_c_pre.cbl.weight[j,:]
                                break
                            else:
                                seen-=1
            else: #for c2
                # get W_c[cid] for each section, have to control concept.txt and concept_init
                for i,con in enumerate(concepts):
                    for j,inter in enumerate(inter_concept_pre): # for the section of task t
                        if(con==inter):
                            #print("assign previous concept:",i)
                            if (args.nonlinear=='False'):
                                proj_layer.weight[i,:]=W_c_pre[j,:]
                            else:
                                proj_layer.cbl.weight[i,:]=W_c_pre.cbl.weight[j,:]
                            break
                    


    opt = torch.optim.Adam(proj_layer.parameters(), lr=5e-4)
    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                if (args.nonlinear=='False'):
                    best_weights = proj_layer.weight.clone()
                else:
                    best_weights=copy.deepcopy(proj_layer)
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                if (args.nonlinear=='False'):
                    best_weights = proj_layer.weight.clone()
                else:
                    best_weights=copy.deepcopy(proj_layer)
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
    
    if (args.nonlinear=='False'):
        proj_layer.load_state_dict({"weight":best_weights})
    else:
        proj_layer=best_weights
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
        
    #delete concepts that are not interpretable (only with if pretrained model)
    if args.pretrain:
        with torch.no_grad():
            outs = proj_layer(val_target_features.to(args.device).detach())
            sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
            interpretable = sim > args.interpretability_cutoff
            
        if args.print:
            for i, concept in enumerate(concepts):
                if sim[i]<=args.interpretability_cutoff:
                    print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
        
        concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    else:
        # if not using pretrained model target features are random and can not be used to 
        # verify if concepts are interpretable. instead make all concepts interpretable
        interpretable = torch.ones(len(concepts), dtype=torch.bool)

    del clip_features, val_clip_features
    if (args.nonlinear=='False'):
        W_c = proj_layer.weight[interpretable]
    else:
        W_c=proj_layer.to('cpu')
        W_c.cbl.weight=torch.nn.Parameter(W_c.cbl.weight[interpretable])

    # CBM Continual Method 1
    if(args.freeze_wc=="True"):
        if(task>0):
            with torch.no_grad():
                # load previous W_c and concept.txt for each task
                # get W_c[cid] for each section, have to control concept.txt and concept_init
                if args.concept_method=="c1":
                    concept_memory={}
                    for i,con in enumerate(concepts):
                        seen=0
                        if(con in concept_memory.keys()):
                            seen=concept_memory[con]
                        for j,inter in enumerate(inter_concept_pre): # for the section of task t
                            if(con==inter):
                                if(seen==0):
                                    if(con in concept_memory.keys()):
                                        concept_memory[con]+=1
                                    else:
                                        concept_memory[con]=1
                                    #print("freeze previous concept:",i)
                                    if (args.nonlinear=='False'):
                                        W_c[i,:]=W_c_pre[j,:]
                                    else:
                                        W_c.cbl.weight[i,:]=W_c_pre.cbl.weight[j,:]
                                    break
                                else:
                                    seen-=1
                else:
                    for i,con in enumerate(concepts):
                        for j,inter in enumerate(inter_concept_pre): # for the section of task t
                            if(con==inter):
                                #print("freeze previous concept:",i)
                                if (args.nonlinear=='False'):
                                    W_c[i,:]=W_c_pre[j,:]
                                else:
                                    W_c.cbl.weight[i,:]=W_c_pre.cbl.weight[j,:]
                                break


    if (args.nonlinear=='False'):
        proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
        proj_layer.load_state_dict({"weight":W_c})
    else:
        proj_layer = torch.nn.Sequential(OrderedDict([
            ('first', torch.nn.Linear(in_features=target_features.shape[1], out_features=4096,bias=True)),
            ('relu1',torch.nn.ReLU()),
            ('cbl', torch.nn.Linear(in_features=4096, out_features=len(concepts),bias=False))
            ]))
        proj_layer.first.weight=W_c.first.weight
        proj_layer.first.bias=W_c.first.bias
        proj_layer.cbl.weight=W_c.cbl.weight
        
    
    train_targets = data_utils.get_targets_only(d_train,args.seed)
    val_targets = data_utils.get_targets_only(d_val,args.seed)
    
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std
        
        train_y = torch.LongTensor(train_targets)
        print("train_c:",train_c.shape)
        print("train_y:",train_y.shape)
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std
        
        val_y = torch.LongTensor(val_targets)

        val_ds = TensorDataset(val_c,val_y)


    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    n_classes=100
    if seg[0] in ['cifar10','imagenet10']:
        n_classes=10
    elif seg[0] in ['CUB200','TinyImagenet']:
        n_classes=200
    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],n_classes).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    if task>0:
        with torch.no_grad():
            w_g_path="{}/{}_cbm/W_g.pt".format(args.save_dir,previous_dataset)
            print("load Wf: ",w_g_path)
            W_g_pre=torch.load(w_g_path).to(args.device)
            if args.concept_method=="c1":
                concept_memory={}
                for i,con in enumerate(concepts):
                    seen=0
                    if(con in concept_memory.keys()):
                        seen=concept_memory[con]
                    for j,inter in enumerate(inter_concept_pre): # for the section of task t
                        if(con==inter):
                            if(seen==0):
                                if(con in concept_memory.keys()):
                                    concept_memory[con]+=1
                                else:
                                    concept_memory[con]=1
                                #print("freeze previous concept's W_f:",i)
                                linear.weight[:,i]=W_g_pre[:,j]
                                break
                            else:
                                seen-=1
            else:
                for i,con in enumerate(concepts):
                    for j,inter in enumerate(inter_concept_pre): # for the section of task t
                        if(con==inter):
                            #print("freeze previous concept's W_f:",i)
                            linear.weight[:,i]=W_g_pre[:,j]
                            break

            b_g_path="{}/{}_cbm/b_g.pt".format(args.save_dir,previous_dataset)
            print("load bf: ",b_g_path)
            b_g_pre=torch.load(b_g_path).to(args.device)
            linear.bias=torch.nn.Parameter(b_g_pre)

    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    if args.solver=="glm":
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                        val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = n_classes)
        W_g = output_proj['path'][0]['weight']
        b_g = output_proj['path'][0]['bias']
    else:
        GAMMA=args.gamma if task>0 else 0
        trainer=Trainer(args,linear,indexed_train_loader,val_loader,args.lam,ALPHA,GAMMA)
        W_g,b_g=trainer.train()
    
    save_name = "{}/{}_cbm".format(args.save_dir, args.dataset)
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if args.solver=='glm':
        with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
            out_dict = {}
            for key in ('lam', 'lr', 'alpha', 'time'):
                out_dict[key] = float(output_proj['path'][0][key])
            out_dict['metrics'] = output_proj['path'][0]['metrics']
            nnz = (W_g.abs() > 1e-5).sum().item()
            total = W_g.numel()
            out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
            json.dump(out_dict, f, indent=2)
    
if __name__=='__main__':
    args = parser.parse_args()
    train_cbm_and_save(args)
    print("Max GPU usage:",torch.cuda.max_memory_allocated('cuda'))
