import os
import math
import numpy as np
import pandas as pd
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import similarity

PM_SUFFIX = {"max":"_max", "avg":""}
DATASET_ROOTS = {
    "imagenet_val": os.path.expanduser("~")+"/OOD_detection/imagenet-r/DeepAugment/ImageNet_val/",
    "broden": "~/dataset/broden1_224/images/"}

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==2:
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==2:
                outputs.append(output.detach())
    return hook

def get_target_model(target_name,ck_dir, device):
    seg=target_name.split("-")
    if seg[0]=='resnet18':
        classes=200 #TinyImagenet
        if seg[2]=='cifar10'or seg[2]=='SplitMNIST':
            classes=10
        elif seg[2]=='cifar100':
            classes=100
        elif seg[2]=='CORe50Dataset':
            classes=50
        elif seg[2]=='CUB200':
            classes=200
        target_model = models.resnet18(pretrained=False, num_classes=classes).to(device)
        #sd='%s%s-%s-%s-task%d-%d.pt' % (args.ck_dir,args.model,args.task,args.dataset,args.task_id,args.task_num)
        sd='%s%s.pt'%(ck_dir,target_name)
        print("probing:",sd)
        target_model.load_state_dict(torch.load(sd))
    elif target_name == 'resnet18_places': #trained on places365
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
    else:
        target_model = eval("models.{}(pretrained=True).to(device)".format(target_name))
        
    target_model.eval()
    return target_model

def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                             PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    all_exist = True
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            all_exist = False
        break
    return all_exist

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}

    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        eval(command)
    
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def save_activations(clip_name, target_name, ck_dir, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, target_preprocess, save_dir,task):
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    target_model = get_target_model(target_name,ck_dir, device)
    #setup data
    data_c, data_t = get_data(d_probe,task, clip_preprocess, target_preprocess)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    save_names = get_save_names(clip_name = clip_name, target_name = target_name,
                                target_layer = '{}', d_probe = d_probe, concept_set = concept_set,
                                pool_mode=pool_mode, save_dir = save_dir)
    target_save_name, clip_save_name, text_save_name = save_names
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    save_target_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode)
    return
    
    
def get_resnet_imagenet_preprocess():
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True,device="cuda"):
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name, map_location='cpu')
    similarity = similarity_fn(clip_feats, target_feats, device=device)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity
    

def get_data(dataset_name,target, clip_preprocess=None, target_preprocess=None):
    if dataset_name == "cifar100_train":
        data_c = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=clip_preprocess)
        data_t = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=target_preprocess)

    elif dataset_name == "cifar100_val":
        data_c = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=clip_preprocess)
        data_t = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=target_preprocess)
        
    elif dataset_name in ("imagenet_val", "broden"):
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
    elif dataset_name =='mnist':
        dataset_c='%s/%s_task%d_%d.pt' % ('~/concept-driven-continual-learning/dataset_dir','SplitFMNIST',target,5)
        dataset_t='%s/%s_1_task%d_%d.pt' % ('~/concept-driven-continual-learning/dataset_dir','SplitFMNIST',target,5)
        print('load dataset 224 saved from training')
        data_c = torch.load(dataset_c)
        data_t = torch.load(dataset_t)
    return data_c, data_t

def get_pil_data(dataset_name):
    if dataset_name == "cifar100_train":
        pil_data= datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True)

    elif dataset_name == "cifar100_val":
        pil_data= datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    elif dataset_name in ("imagenet_val", "broden"):
        root = DATASET_ROOTS[dataset_name]
        pil_data= datasets.ImageFolder(root)
        
    elif dataset_name == "imagenet_broden":
        root_i = DATASET_ROOTS["imagenet_val"]
        root_b = DATASET_ROOTS["broden"]
        pil_data= torch.utils.data.ConcatDataset([datasets.ImageFolder(root_i), 
                                                  datasets.ImageFolder(root_b)])
    return pil_data


def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]
    
    return cifar100_has_superclass, cifar100_doesnt_have_superclass

def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")
    
    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')
        
        found = (name+'-s' in broden_scenes['name'].values)
        
        if found:
            id_to_broden_label[i] = name.replace('-', '/')+'-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

    
    
