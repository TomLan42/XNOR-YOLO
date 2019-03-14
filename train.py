import argparse
import os 
import shutil
import sys
import gc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.
import torch.utils.data.distributed

import model_list
import util





def train(train_loader,model,criterion,optimizer,epoch):
    losses = AverageMeter()

    model.train()

    for i , (input, target) in enumerate(train_loader):

        #Mesure data loading time.
        data_time.update(time.time() - end)

        target = target.cuda(async = True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        #Weight Binarization.
        bin_op.binariztion()
        
        #Forward propagation and compute yolo loss.
        ouput = model(input_var)
        loss = criterion(output, target_var)

        
        #record loss


        #Computer gradient.
        optimizer.zero_grad()
        loss.backward()


        #Restore binarized weight to full precision and update.
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        optimizer.step()
        

        #Print losses etc.
        
    
        gc.collect()



def main():
    
    model = model_list.alexnet(pretrained=args.pretrained)
    criterion = yoloLoss(7,2,5,0.5)
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    '''Data loading module'''
    train_dataset = yoloDataset(root=file_root,
        list_file=['voc2012.txt','voc2007.txt'],train=True,transform = [transforms.ToTensor()] )
    train_loader = DataLoader(
        train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    

    test_dataset = yoloDataset(root=file_root,
        list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
    
    test_loader = DataLoader(
        test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)


