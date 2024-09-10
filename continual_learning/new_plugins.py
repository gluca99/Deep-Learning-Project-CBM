'''
define new plugins here
direction 1 and direction 2
Not sure whether it's necessary to define new strategies of just use Naive
  -> if new need to train on datasets besides current experiment

Using EWC plugin as template
'''
from collections import defaultdict
from typing import Dict, Tuple
import warnings
import itertools
import random
import os,csv
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch import linalg as LA

import subprocess,json,csv

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict


class EWC_BASE_Plugin(SupervisedPlugin):
    """
    Elastic Weight Consolidation (EWC) plugin.
    EWC computes importance of each weight at the end of training on current
    experience. During training on each minibatch, the loss is augmented
    with a penalty which keeps the value of the current weights close to the
    value they had on previous experiences in proportion to their importance
    on that experience. Importances are computed with an additional pass on the
    training set. This plugin does not use task identities.
    """

    def __init__(
            self,
            ewc_lambda,
            #new strategy
            model, dataset,
            alpha,beta,gamma,
            mode="separate",
            decay_factor=None,
            keep_importance_data=False,
            # new strategy
            task_num=5,ck_dir='./ck_dir/',
            result_dir='./results',dataset_dir='./dataset_dir/',probing_broden=False
    ):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience.
               `online` to keep a single penalty summed with a decay factor
               over all previous tasks.
        :param decay_factor: used only if mode is `online`.
               It specifies the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        """

        super().__init__()
        assert (decay_factor is None) or (
                mode == "online"
        ), "You need to set `online` mode to use `decay_factor`."
        assert (decay_factor is not None) or (
                mode != "online"
        ), "You need to set `decay_factor` to use the `online` mode."
        assert (
                mode == "separate" or mode == "online"
        ), "Mode must be separate or online."

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor
        # new strategy
        self.model=model
        self.dataset=dataset
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.task_num=task_num
        self.ck_dir=ck_dir
        self.result_dir=result_dir
        self.dataset_dir=dataset_dir
        self.probing_broden=probing_broden

        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == "separate":
            for experience in range(exp_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        strategy.model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience],
                ):
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    n_units = saved_param.shape[0]
                    cur_param = cur_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == "online":
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp],
            ):
                # dynamic models may add new units
                # new units are ignored by the regularization
                n_units = saved_param.shape[0]
                cur_param = cur_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        strategy.loss += self.ewc_lambda * penalty

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        exp_counter = strategy.clock.train_exp_counter
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
            exp_counter, # new strategy
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(strategy.model)
        21784587# clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def compute_importances(
            self, model, criterion, optimizer, dataset, device, batch_size,exp
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)

        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") \
            else None
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                collate_fn=collate_fn)
        
        # new strategy
        model_save_path='%s%s-%s-%s-task%d-%d.pt' % (self.ck_dir,self.model,'EWC_based',self.dataset,exp,self.task_num)
        torch.save(model.state_dict(), model_save_path)
        #iou_list=self.get_iou_score(exp)
        IOU_THRESHOLD=4
        for j, (k2, imp) in enumerate(importances):
          name=k2.split(".")
          # before concept unit -> fixed
          if self.model=='alexnet':
            if name[0] in ['conv1','conv2','conv3','conv4']:
              importances[j]=(importances[j][0],(torch.full(imp.shape,self.alpha)).to(device))
            
            # layer where concept units belong
            elif k2== 'conv5.weight':
              imp_list=[]
              print("*********** fix concept units ***********")
              for (i,iou) in enumerate(iou_list):
                if float(iou)>=IOU_THRESHOLD:
                  print("conv5-%d: %.4f"%(i,float(iou)))
                  unit_imp=torch.full(imp[i].shape,self.alpha)
                else:
                  unit_imp=torch.full(imp[i].shape,self.gamma)
                imp_list.append(unit_imp)
              
              importances[j]=(importances[j][0],(torch.stack(imp_list)).to(device))
            # fc layer, modified ith column if ith unit is interpretable
              '''
              elif k2=='fc6.weight':
                for (i,iou) in enumerate(iou_list):
                  if float(iou)>=IOU_THRESHOLD:
                    imp[:,36*i]=self.beta
                    print("fc6-%d: %.4f"%(i,float(iou)))
                    importances[j]=(importances[j][0],imp.to(device))
              '''
            else: # after fc6 weight layer
              importances[j]=(importances[j][0],(torch.full(imp.shape,self.gamma)).to(device))
          else: # resnet implementation
              if (name[0] == 'fc') or (name[0]== 'avgpool'):
                importances[j]=(importances[j][0],(torch.full(imp.shape,self.gamma)).to(device))
              '''
              if (name[0] in ['layer1','layer2','layer3','layer4']) or (name[0]=='layer4' and name[1]=='0') or (k2 in ['layer4.1.conv1.weight','layer4.1.bn1.weight','layer4.1.bn1.bias']):
                importances[j]=(importances[j][0],(torch.full(imp.shape,self.alpha)).to(device))
            
              # layer where concept units belong
              elif k2== 'layer4.1.conv2.weight':
                imp_list=[]
                print("*********** fix concept units ***********")
                for (i,iou) in enumerate(iou_list):
                  if float(iou)>=IOU_THRESHOLD:
                    print("conv5-%d: %.4f"%(i,float(iou)))
                    unit_imp=torch.full(imp[i].shape,self.alpha)
                  else:
                    unit_imp=torch.full(imp[i].shape,self.gamma)
                  imp_list.append(unit_imp)
                
                importances[j]=(importances[j][0],(torch.stack(imp_list)).to(device))
              # fc layer, modified ith column if ith unit is interpretable
                elif k2=='fc.weight':
                  for (i,iou) in enumerate(iou_list):
                    if float(iou)>=IOU_THRESHOLD:
                      imp[:,i]=self.beta
                      print("fc-%d: %.4f"%(i,float(iou)))
                      importances[j]=(importances[j][0],imp.to(device))
              else:
                importances[j]=(importances[j][0],(torch.full(imp.shape,self.gamma)).to(device))
              '''

        # average over mini batch length
        for j,(_, imp) in enumerate(importances):
            importances[j] = (importances[j][0],(importances[j][1]/float(len(dataloader))).to(device))

        return importances
    
    def get_iou_score(self,exp):
      '''
        exec NetDissect
        return iou score in a list :list(iou) , len = number of units in that layer

        in further exp, return Dict[layer]:list(iou) 
      '''
      if self.dataset=='cifar100':
        num_classes=100
      elif self.dataset=='TinyImagenet':
        num_classes=200
      elif self.dataset=='CLEAR':
        num_classes=11
      else:
        num_classes=10 
      
      if self.model=='alexnet':
        layer='conv5'
      else:
        layer='layer4'

      if self.probing_broden==False:
        ROOT='./experiment/'
        dissect_script=ROOT+'dissect_experiment.py'
        command=['python',dissect_script,'--model',self.model,'--dataset',self.dataset, '--layer',layer, # change layer if need more NetDissect
                '--task_num',str(self.task_num),'--task','EWC_based','--task_id',str(exp),
                '--target',str(exp),'--num_classes',str(num_classes),'--ck_dir',self.ck_dir,
                '--result_dir',self.result_dir,'--dataset_dir',self.dataset_dir]
        subprocess.call(command,cwd=ROOT)

        iou_report='%s/%s-%s-%s-target%d-task%d-%d-%s-10/report.json' % (self.result_dir,self.model, self.dataset,'EWC_based',exp,exp,self.task_num,layer)
        with open(iou_report, 'r') as f:
          report = json.load(f)

        iou_list=[]
        for d in report['units']:
            iou_list.append(d['iou'])
      else:
        ROOT='./NetDissect-Lite/'
        setting=ROOT+'main.py'
        command=['python',setting,'--model',self.model,'--dataset',self.dataset, '--layer',layer, # change layer if need more NetDissect
                '--task_num',str(self.task_num),'--task','EWC_based','--task_id',str(exp),
                '--target',str(exp),'--num_classes',str(num_classes),'--ck_dir',self.ck_dir,
                '--result_dir',self.result_dir]
        subprocess.call(command,cwd=ROOT)
        iou_report='%s/%s-%s-%s-target%d-task%d-%d-%s/tally.csv' % (self.result_dir,self.model, self.dataset,'EWC_based',exp,exp,self.task_num,layer)
        with open(iou_report, newline='') as f:
            report = csv.reader(f)
            if self.model=='alexnet':
              unit_num=256
            else:
              unit_num=512
            iou_list=[0]*unit_num
            for i,row in enumerate(report):
                if(i==0):
                    continue
                iou_list[int(row[0])-1]=float(row[3])

      return iou_list 

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                    self.importances[t - 1], importances,
                    fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    self.importances[t].append((k2, curr_imp))
                    continue

                assert k1 == k2, "Error in importance computation."

                self.importances[t].append(
                    (k1, (self.decay_factor * old_imp + curr_imp))
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


ParamDict = Dict[str, Tensor]
EwcDataType = Tuple[ParamDict, ParamDict]



''''
#####################
Selective Retraining, freeze subnetwork of the concept units
#####################
'''
class my_hook(object):

    def __init__(self, mask1, mask2):
        self.mask1 = torch.Tensor(mask1).long().nonzero().view(-1).numpy()
        self.mask2 = torch.Tensor(mask2).long().nonzero().view(-1).numpy()

    def __call__(self, grad):
        # if prev_active[x]=True, x's gradient will be zero      
        grad_clone = grad.clone()
        if self.mask1.size:
            grad_clone[self.mask1, :,:,:] = 0
        if self.mask2.size:
            grad_clone[:, self.mask2,:,:] = 0
        return grad_clone

class my_hook_sol1(object):

    def __init__(self, mask1):
        self.mask1 = torch.Tensor(mask1).long().nonzero().view(-1).numpy()

    def __call__(self, grad):
        # if prev_active[x]=True, x's gradient will be zero      
        grad_clone = grad.clone()
        if self.mask1.size:
            grad_clone[self.mask1, :,:,:] = 0
        return grad_clone

class hook_bn(object):

    def __init__(self, mask1):
        self.mask1 = torch.Tensor(mask1).long().nonzero().view(-1).numpy()

    def __call__(self, grad):
        # if prev_active[x]=True, x's gradient will be zero      
        grad_clone = grad.clone()
        if self.mask1.size:
            grad_clone[self.mask1] = 0
        return grad_clone

class SRT_Plugin(SupervisedPlugin):

    def __init__(
            self,
            #new strategy
            model, dataset,
            l1_coeff,zero_threshold,probing_method,random,sol,n_classes,strategy,
            task_num=5,ck_dir='./ck_dir/',
            result_dir='./results',dataset_dir='./dataset_dir/',probing_broden=False
    ):

        super().__init__()

        # new strategy
        self.model=model
        self.dataset=dataset
        self.l1_coeff=l1_coeff
        self.zero_threshold=zero_threshold
        self.task_num=task_num
        self.ck_dir=ck_dir
        self.result_dir=result_dir
        self.strategy=strategy
        self.dataset_dir=dataset_dir
        self.probing_method=probing_method
        self.random=random
        self.sol=sol
        self.n_classes=n_classes
        self.prev=0
        self.prev_0=0


    def before_backward(self, strategy, **kwargs):
        # Loss with Norm penalty, to make NN sparse enough
        penalty = 0
        for name, param in strategy.model.named_parameters():
            if param.requires_grad and ('fc' not in name):
              if param.dim()==4:  
                reg=LA.norm(param, dim=(2,3))
                reg=reg.norm(1)
              else:
                reg=param.norm(1)
              penalty = penalty + reg

        strategy.loss+= self.l1_coeff * penalty
    
    def before_forward(self, strategy, **kwargs):
        strategy.optimizer.zero_grad(set_to_none=True)
    
    def after_training_exp(self, strategy, **kwargs):
        # Save model, do NetDissect
        exp_counter = strategy.clock.train_exp_counter
        model_save_path='%s%s-%s-%s-task%d-%d.pt' % (self.ck_dir,self.model,self.strategy,self.dataset,exp_counter,self.task_num)
        torch.save(strategy.model.state_dict(), model_save_path)
        concept_list=self.netdissect(exp_counter)
        
        # Find subnetwork of the concept units
        with torch.no_grad():
          layers = []
          # only pick conv layers
          for name, param in strategy.model.named_parameters():
              if ('fc' not in name):
                  layers.append(param)
                  print(name)
          layers = reversed(layers)

          prev_active=concept_list
          hooks = []
          selected = []

          # handle downsample for resnet18
          downsample_detector=0
          i=0      
          for layer in layers:
              i+=1
              if(i==1):
                if(strategy.clock.train_exp_counter==0):
                    self.prev_0=layer.detach().clone()
                else:
                    print("**************equal_0:",torch.equal(self.prev_0,layer),layer.size())
              if(i==3):
                if(strategy.clock.train_exp_counter==0):
                    self.prev=layer.detach().clone()
                else:
                    print("**************equal:",torch.equal(self.prev,layer),layer.size())
                    self.prev=layer.detach().clone()
              if (layer.dim()==1):
                h = layer.register_hook(hook_bn(prev_active))
                selected.append( (sum(prev_active), len(prev_active)) )
              else:
                x_size= layer.size()[0]
                y_size= layer.size()[1]
                active = [False]*y_size
                data = layer.data
                for x in range(x_size):

                    # we skip the weight if connected neuron wasn't selected
                    if prev_active[x]==False:
                        continue

                    for y in range(y_size):
                        weight = data[x,y]
                        #print(weight.dim())
                        norm=weight
                        if(norm.dim()!=0):
                          norm=LA.norm(norm)
                        # check if weight is active
                        if (norm > self.zero_threshold):
                            # mark connected neuron as active
                            active[y] = True

                # make parameter freeze
                if self.sol=='sol0':     
                  h = layer.register_hook(my_hook(prev_active, active))
                else:
                  h = layer.register_hook(my_hook_sol1(prev_active))
              
                if (self.model=='alexnet'):
                  prev_active = active
                else: # for Resnet
                  if (x_size!=y_size):
                    if(downsample_detector==0):
                      downsample_list=active
                      downsample_detector+=1
                    else:
                      prev_active=[a or b for a, b in zip(downsample_list, active)]
                      downsample_detector=0
                  else:
                    prev_active = active
                
                selected.append( (sum(active), y_size) )
              # end else
              hooks.append(h)
              
          if (self.random==False):
            freeze_dir="%s/freeze_stat" %(self.result_dir)
          else:
            freeze_dir="%s/freeze_random_stat" %(self.result_dir)
          
          if not os.path.exists(freeze_dir):
            os.makedirs(freeze_dir)
          freeze_file="%s/freeze_task%d.csv"%(freeze_dir,exp_counter)

          with open(freeze_file,mode="w",newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(['layer','freeze', 'total'])
            for nr, (sel, neurons) in enumerate(reversed(selected)):
                print( "Freeze layer %d: %d / %d" % (nr+1, sel, neurons) )
                writer.writerow([nr+1,sel,neurons])
    
    def netdissect(self,exp):
      '''
        exec NetDissect
        return iou score in a list :list(concept) , len = number of units in that layer

        in further exp, return Dict[layer]:list(concept) 
      '''
      if self.dataset=='cifar100':
        num_classes=100
      elif self.dataset=='TinyImagenet':
        num_classes=200
      elif self.dataset=='CLEAR':
        num_classes=11
      elif self.dataset=='cifar10':
        num_classes=10 
      elif self.dataset=='imagenet10':
        num_classes=10 
      elif self.dataset=='CORe50Dataset':
        num_classes=self.n_classes
      elif self.dataset=='CUB200':
        num_classes=self.n_classes
      elif self.dataset=='SplitMNIST':
        num_classes=10
      
      if self.model=='alexnet':
        layer='conv5'
      else:
        layer='layer4'
      
      target=exp
      IOU_THRESHOLD=0.04
      if self.probing_method=="NetDissect":
        ROOT='./experiment/'
        dissect_script=ROOT+'dissect_experiment.py'
        command=['python',dissect_script,'--model',self.model,'--dataset',self.dataset, '--layer',layer, # change layer if need more NetDissect
                '--task_num',str(self.task_num),'--task','SRT','--task_id',str(exp),
                '--target',str(target),'--num_classes',str(num_classes),'--ck_dir',self.ck_dir,
                '--result_dir',self.result_dir,'--dataset_dir',self.dataset_dir]
        subprocess.call(command,cwd=ROOT)

        iou_report='%s/%s-%s-%s-target%d-task%d-%d-%s-10/report.json' % (self.result_dir,self.model, self.dataset,'SRT',target,exp,self.task_num,layer)
        with open(iou_report, 'r') as f:
          report = json.load(f)
        
        if self.random==False:
          concept_list=[]
          for d in report['units']:    
              if float(d['iou'])>=IOU_THRESHOLD:
                  concept_list.append(True)
              else:
                concept_list.append(False)
        else:
          concept_list=[False]*512
          num=0
          for d in report['units']:
              if float(d['iou'])>=IOU_THRESHOLD:
                  num+=1
          id=random.sample(range(512),num)
          for i in id:
            concept_list[i]=True

      elif self.probing_method=="CLIP-Dissect":
        IOU_THRESHOLD=0.3
        ROOT='./sandbox-clip-dissect-main/'
        dissect_script=ROOT+'describe_neurons.py'
        model ="%s-%s-%s-task%d-%d"%(self.model,self.strategy,self.dataset,exp,self.task_num)
        #if self.dataset=='SplitFMNIST':
        #  command=['python',dissect_script,'--target_model',model,'--ck_dir',self.ck_dir,'--activation_dir',self.result_dir,'--result_dir',self.result_dir,'--task',str(target),'--d_probe','mnist']
        #else:
        command=['python',dissect_script,'--target_model',model,'--ck_dir',self.ck_dir,'--activation_dir',self.result_dir,'--result_dir',self.result_dir]
          # for testing time 
          #command=['python',dissect_script,'--target_model',model,'--ck_dir',self.ck_dir,'--result_dir',self.result_dir]
        if self.random==False:
          subprocess.call(command,cwd=ROOT)

        if self.model=='alexnet':
          unit_num=256
        elif self.model=='resnet50':
          unit_num=2048
        else: #resnet18,34
          unit_num=512
        
        iou_report='%s/%s_all.csv' % (self.result_dir,model)
        with open(iou_report, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        
        if self.random==False:
          concept_list=[]
          for line in data[1:]:
            if (float(line[3])>=IOU_THRESHOLD):
              concept_list.append(True)
            else:
              concept_list.append(False)
        else:
          concept_list=[False]*unit_num
          num=0
          for line in data[1:]:
            if (float(line[3])>=IOU_THRESHOLD):
              num+=1
          id=random.sample(range(unit_num),num)
          for i in id:
            concept_list[i]=True

      else:
        ROOT='./NetDissect-Lite/'
        setting=ROOT+'main.py'
        command=['python',setting,'--model',self.model,'--dataset',self.dataset, '--layer',layer, # change layer if need more NetDissect
                '--task_num',str(self.task_num),'--task','SRT','--task_id',str(exp),
                '--target',str(target),'--num_classes',str(num_classes),'--ck_dir',self.ck_dir,
                '--result_dir',self.result_dir]
        subprocess.call(command,cwd=ROOT)
        iou_report='%s/%s-%s-%s-target%d-task%d-%d-%s/tally.csv' % (self.result_dir,self.model, self.dataset,'SRT',target,exp,self.task_num,layer)
        with open(iou_report, newline='') as f:
            report = csv.reader(f)
            concept_list=[]
            for i,row in enumerate(report):
                if(i==0):
                    continue
                if float(row[3])>=IOU_THRESHOLD:
                    concept_list.append(True)
                else:
                  concept_list.append(False)

      return concept_list


class CBM_Plugin(SupervisedPlugin):
    def __init__(
            self,
            ewc_lambda,sparse_lam,
            #new strategy
            model, dataset,
            seed,gamma,dif,nonlinear,task_num,replay='gem',
            mode="separate",
            decay_factor=None,
            keep_importance_data=False,
            # new strategy
            ck_dir='./ck_dir/',
            result_dir='./results',dataset_dir='./dataset_dir/',probing_broden=False
    ):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience.
               `online` to keep a single penalty summed with a decay factor
               over all previous tasks.
        :param decay_factor: used only if mode is `online`.
               It specifies the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        """

        super().__init__()
        assert (decay_factor is None) or (
                mode == "online"
        ), "You need to set `online` mode to use `decay_factor`."
        assert (decay_factor is not None) or (
                mode != "online"
        ), "You need to set `decay_factor` to use the `online` mode."
        assert (
                mode == "separate" or mode == "online"
        ), "Mode must be separate or online."

        self.ewc_lambda = ewc_lambda
        self.sparse_lam = sparse_lam
        self.mode = mode
        self.decay_factor = decay_factor
        # new strategy
        self.model=model
        self.dataset=dataset
        self.seed=seed
        self.gamma=gamma
        self.dif=dif
        self.nonlinear=nonlinear
        self.task_num=task_num
        self.replay=replay
        self.ck_dir=ck_dir
        self.result_dir=result_dir
        self.dataset_dir=dataset_dir
        self.probing_broden=probing_broden

        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        LAM=self.sparse_lam
        ALPHA=0.99
        strategy.loss += LAM * ALPHA * strategy.model.final.weight.norm(p=1) + 0.5 * LAM * (1 - ALPHA) * (strategy.model.final.weight**2).sum()
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)
        if self.mode == "separate":
            for experience in range(exp_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        strategy.model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience],
                ):
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    n_units = saved_param.shape[0]
                    cur_param = cur_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == "online":
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp],
            ):
                # dynamic models may add new units
                # new units are ignored by the regularization
                n_units = saved_param.shape[0]
                cur_param = cur_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        strategy.loss += self.ewc_lambda * penalty

    def before_training_exp(self, strategy, **kwargs):
        exp_counter = strategy.clock.train_exp_counter
        freeze=False
        if self.replay=='gem':        
          naive_path=self.result_dir.replace("CBM_GEM","cc_cbm")
        elif self.replay=='der':
            naive_path=self.result_dir.replace("CBM_DER","cc_cbm")
        elif self.replay=='gdumb':        
          naive_path=self.result_dir.replace("freeze_reg_gdumb","freeze_reg")
        elif self.replay=='mir':        
          freeze=False
          naive_path=self.result_dir.replace("CBM_MIR","cc_cbm")

        if self.dif!='False':
          naive_path=naive_path.replace("/dif","")
        w_c_path='%s/%s_task_%d_%d_%s_cbm/W_c.pt' % (naive_path,self.dataset,exp_counter,self.task_num,self.seed)
        mean_path='%s/%s_task_%d_%d_%s_cbm/proj_mean.pt' % (naive_path,self.dataset,exp_counter,self.task_num,self.seed)
        std_path='%s/%s_task_%d_%d_%s_cbm/proj_std.pt' % (naive_path,self.dataset,exp_counter,self.task_num,self.seed)
        concept_path='%s/%s_task_%d_%d_%s_cbm/concepts.txt' % (naive_path,self.dataset,exp_counter,self.task_num,self.seed)
        inter_concept_path='%s/%s_task_%d_%d_%s_cbm/concepts.txt' % (naive_path,self.dataset,self.task_num-1,self.task_num,self.seed)
        print("load Naive Wc: ",w_c_path)

        W_c_pre=torch.load(w_c_path,map_location=strategy.device)
        if self.nonlinear=='False':
          if freeze == True:
             W_c_pre.requires_grad=False 
        else:
          strategy.model.proj_layer.first.weight=W_c_pre.first.weight
          strategy.model.proj_layer.first.bias=W_c_pre.first.bias
          if freeze == True:
            W_c_pre.first.weight.requires_grad=False
            W_c_pre.first.bias.requires_grad=False
            W_c_pre.cbl.weight.requires_grad=False
        mean_pre=torch.load(mean_path,map_location=strategy.device)
        std_pre=torch.load(std_path,map_location=strategy.device)
        if freeze==True:
            mean_pre.requires_grad=False
            std_pre.requires_grad=False
        print("load concept.txt: ",concept_path)
        with open(inter_concept_path) as f:
            inter_concept_pre=f.read().split("\n")
        with open(concept_path) as f:
            concepts=f.read().split("\n")
        # concept set method 1
        concept_memory={}
        with torch.no_grad():
          for i,con in enumerate(inter_concept_pre):
              seen=0
              if(con in concept_memory.keys()):
                  seen=concept_memory[con]
              for j,inter in enumerate(concepts): # for the section of task t
                  if(con==inter):
                      if(seen==0):
                          if(con in concept_memory.keys()):
                              concept_memory[con]+=1
                          else:
                              concept_memory[con]=1
                          #print("assign previous concept:",i)
                          if self.nonlinear=='False':
                            strategy.model.proj_layer.weight[i,:]=W_c_pre[j,:]
                          else:
                            strategy.model.proj_layer.cbl.weight[i,:]=W_c_pre.cbl.weight[j,:]
                          strategy.model.proj_mean[:,i]=mean_pre[:,j]
                          strategy.model.proj_std[:,i]=std_pre[:,j]
                          break
                      else:
                          seen-=1
    
    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        exp_counter = strategy.clock.train_exp_counter
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
            exp_counter, # new strategy
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(strategy.model)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def compute_importances(
            self, model, criterion, optimizer, dataset, device, batch_size,exp
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)

        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") \
            else None
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                collate_fn=collate_fn)
        
        IOU_THRESHOLD=4
        for j, (k2, imp) in enumerate(importances):
          name=k2.split(".")
          if (name[0] == 'wf'):
            importances[j]=(importances[j][0],(torch.full(imp.shape,self.gamma)).to(device))

        # average over mini batch length
        for j,(_, imp) in enumerate(importances):
            importances[j] = (importances[j][0],(importances[j][1]/float(len(dataloader))).to(device))

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            print('update_importances:',len(importances[0]))
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                    self.importances[t - 1], importances,
                    fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    self.importances[t].append((k2, curr_imp))
                    continue

                assert k1 == k2, "Error in importance computation."

                self.importances[t].append(
                    (k1, (self.decay_factor * old_imp + curr_imp))
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")

    '''
    def update_importances(self, importances, t: int):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[t - 1].items(),
                importances.items(),
                fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    assert k2 is not None
                    assert curr_imp is not None
                    self.importances[t][k2] = curr_imp
                    continue

                assert k1 == k2, "Error in importance computation."
                assert curr_imp is not None
                assert old_imp is not None
                assert k2 is not None

                # manage expansion of existing layers
                self.importances[t][k1] = ParamData(
                    f"imp_{k1}",
                    curr_imp.shape,
                    init_tensor=self.decay_factor * old_imp.expand(curr_imp.shape)
                    + curr_imp.data,
                    device=curr_imp.device,
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")
        '''

class SPARSE_Plugin(SupervisedPlugin):

    def __init__(
            self
    ):

        super().__init__()

    def before_backward(self, strategy, **kwargs):
        """
        Compute Sparse penalty and add it to the loss.
        """
        LAM=0.00007
        ALPHA=0.99
        for totname, cur_param in strategy.model.named_parameters():
            name=totname.split(".")
            if (name[0] == 'fc'):
                strategy.loss += LAM * ALPHA * cur_param.norm(p=1) + 0.5 * LAM * (1 - ALPHA) * (cur_param**2).sum()
