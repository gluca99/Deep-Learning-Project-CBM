import torch
import os
import json
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from sklearn.metrics import classification_report

def evaluation(outputs, labels):
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

class Model(nn.Module):
    def __init__(self, linear,lam,alpha,gamma,ori_w,ori_b):
        super(Model, self).__init__()
        self.classifier=linear
        self.loss=nn.CrossEntropyLoss()
        self.lam=lam
        self.alpha=alpha
        self.gamma=gamma
        self.original_model_weight=ori_w
        self.original_model_bias=ori_b

    def forward(self,x):
        y=self.classifier(x)
        return y

    def cal_loss(self,pred,target):
        loss=self.loss(pred,target)
        loss+=self.lam * self.alpha * self.classifier.weight.norm(p=1) + 0.5 * self.lam * (1 - self.alpha) * (self.classifier.weight**2).sum()
        loss+=self.gamma * ((self.classifier.weight-self.original_model_weight)**2).sum()+self.gamma*((self.classifier.bias-self.original_model_bias)**2).sum()
        return  loss
    
class Trainer:
    def __init__(self,args,linear,tr_set,dev_set,lam,alpha,gamma):
        self.args = args
        self.model=linear
        self.tr_set=tr_set
        self.tr_size=len(tr_set.dataset)
        self.dev_set=dev_set
        self.dev_size=len(dev_set.dataset)
        self.lam=lam
        self.alpha=alpha
        self.gamma=gamma
        self.best_w=None
        self.best_b=None

    def train(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)

        epoch_pbar = trange(self.args.num_epoch, desc="Epoch")
        best_acc=0
        original_model_weight=self.model.weight.detach().clone()
        original_model_bias=self.model.bias.detach().clone()
        self.model=Model(self.model,self.lam,self.alpha,self.gamma,original_model_weight,original_model_bias)
        for epoch in epoch_pbar:
            # Training loop - iterate over train dataloader and update model weights
            self.model.train()
            total_loss,train_acc=0,0
            for batch in self.tr_set:
                x=batch[0]
                y=batch[1]
                optimizer.zero_grad()
                x,y=x.to(self.args.device),y.to(self.args.device)
                pred=self.model(x)
                _, label= torch.max(pred,1)
                y=y.to(torch.long)
                loss = self.model.cal_loss(pred, y)  # compute loss
                loss.backward()                 # compute gradient (backpropagation)
                optimizer.step()                    # update model with optimizer
                correct = evaluation(label, y) # calculate training accuracy 
                train_acc += correct
                total_loss += loss.item()

            train_info_json = {"epoch": epoch,"train_loss": total_loss/self.tr_size,"train_Acc":train_acc/self.tr_size}
            print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

            self.model.eval()                                # set model to evalutation mode
            total_acc=0
            ans_list=[]
            preds=[]
            with torch.no_grad():
                for batch in self.dev_set: 
                    x=batch[0]
                    y=batch[1]                     
                    for ans_label in y:
                        ans_label=int(ans_label)
                        ans_list.append(ans_label)
                    x,y=x.to(self.args.device),y.to(self.args.device)
                    pred=self.model(x)
                    _, label= torch.max(pred,1)
                    y=y.to(torch.long)
                    correct=evaluation(label,y)
                    total_acc+=correct
                    for p in label.cpu().numpy():
                        preds.append(p)
            
            #print(classification_report(ans_list, preds))
                                                                                                                                                   
            valid_info_json = {"epoch": epoch,"val_Acc":total_acc/self.dev_size*100}
            print(f"{'#' * 30} VALID: {str(valid_info_json)} {'#' * 30}")

            if total_acc > best_acc:
                best_acc = total_acc
                self.best_w=self.model.classifier.weight.detach().clone()
                self.best_b=self.model.classifier.bias.detach().clone()
                print('saving model with acc {:.3f}\n'.format(total_acc/self.dev_size*100))

        return self.best_w,self.best_b
