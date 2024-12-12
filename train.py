'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts 
from datasets.prostate import ProstateDataset 
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import time
from utils.logger import log
import os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from patient_info.patient_tensor_get import Clinic_Info
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import Compose, RandRotate90, GaussianSmooth, RandGaussianNoise, RandAffine, RandScaleIntensity, RandBiasField, Rand3DElastic, RandZoom, RandSpatialCrop, RandFlip, OneOf
# from utils.visualizer import TrainingVisualizer
from utils.focal_loss import FocalLoss
import matplotlib.pyplot as plt
import pandas as pd
from monai.utils import set_determinism

LOSS = []


class LogColors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'

def caculate_metrics(all_labels, all_predictions):
    accuracy = accuracy_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    
    log.info(
            f'{LogColors.GREEN}roc_auc: = {roc_auc}, accuracy = {accuracy}, sensitivity = {sensitivity}, specificity = {specificity}{LogColors.RESET}')
    return roc_auc, accuracy, sensitivity, specificity

def validate(val_data_loader, model, writer, global_val_step, sets):
    model.eval()
    predicts = []
    gts = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(val_data_loader):       
            if sets.mixed_clinic:
                volumes_ct, volumes_pet, class_array, patient_name, modal_list = batch_data
                tensor_patient = Clinic_Info(sets.excel_path, sets.json_path, modal_list[0])
                tensor_clinic = tensor_patient.process_info(patient_name)
            else:
                volumes_ct, volumes_pet, class_array, _, _ = batch_data

            if not sets.no_cuda: 
                volumes_ct = volumes_ct.cuda()
                volumes_pet = volumes_pet.cuda()
                class_array = class_array.cuda()
                if sets.mixed_clinic:
                    tensor_clinic = tensor_clinic.cuda()
            if sets.mixed_clinic:
                outputs = model(volumes_ct, volumes_pet, tensor_clinic, modal='val')
            else:
                outputs = model(volumes_ct, volumes_pet, tensor_clinic = None)
            outputs_logit = outputs.argmax(dim=1) 
            predicts.append(outputs_logit.cpu().detach().numpy())
            gts.append(class_array.cpu().detach().numpy())
            

    predicts = np.concatenate(predicts).flatten().astype(np.int16)
    gts = np.concatenate(gts).flatten().astype(np.int16)
    roc_auc, accuracy, sensitivity, specificity = caculate_metrics(gts, predicts)
    writer.add_scalar('roc_auc', roc_auc, global_val_step)
    writer.add_scalar('accuracy', accuracy, global_val_step)
    writer.add_scalar('sensitivity', sensitivity, global_val_step)
    writer.add_scalar('specificity', specificity, global_val_step)
    
    return roc_auc, accuracy, sensitivity, specificity

def train(train_data_loader, 
          model, 
          optimizer, 
          scheduler, 
          epoch,
          loss_class,
          writer,
          global_step,
          sets):
    batches_per_epoch = len(train_data_loader)
    
    model.train()
    scheduler.step()
    log.info('lr = {}'.format(scheduler.get_lr())) 
    
    for batch_id, batch_data in enumerate(train_data_loader):
        batch_id_sp = epoch * batches_per_epoch
        if sets.mixed_clinic: 
            volumes_ct, volumes_pet, class_array, patient_name, modal_list = batch_data
            tensor_patient = Clinic_Info(sets.excel_path, sets.json_path, modal_list[0])
            tensor_clinic = tensor_patient.process_info(patient_name)
        else:
            volumes_ct, volumes_pet, class_array, _, _ = batch_data

        if not sets.no_cuda: 
            volumes_ct = volumes_ct.cuda()
            volumes_pet = volumes_pet.cuda()
            class_array = class_array.cuda()
            if sets.mixed_clinic:
                tensor_clinic = tensor_clinic.cuda()
        optimizer.zero_grad()
        if sets.mixed_clinic:
            out_classification = model(volumes_ct, volumes_pet, tensor_clinic, modal='train')
        else:
            out_classification = model(volumes_ct, volumes_pet, tensor_clinic = None)
            
        loss_value_class = loss_class(out_classification, class_array)
        loss = loss_value_class
        if sets.pretrain_path:
            current_lr = optimizer.param_groups[1]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
            
        global_step = global_step + 1
        LOSS.append(loss.item())
        plt.plot(LOSS)
        plt.xlabel('Iters')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.savefig('...')
        writer.add_scalar('Loss', loss.item(), global_step)
        writer.add_scalar('Learning Rate', current_lr, global_step)
        
        loss.backward()                
        optimizer.step()
    return batch_id, batch_id_sp, loss.item(), loss_value_class.item()

def main(train_data_loader, val_data_loader, model, optimizer, scheduler, total_epochs, save_folder, sets):

    if sets.focalloss:
        loss_seg = FocalLoss(sets.n_seg_classes)
    elif sets.regualarization:
        weights = torch.tensor([sets.weight_for_negative_class, sets.weight_for_negative_class - 1])
        loss_seg = nn.CrossEntropyLoss(weight = weights, ignore_index=-1)
    else:
        loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
        
    count = 0

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()
    
    best_auc = 0.0
    log.info('{} epochs in total'.format(total_epochs))
    writer = SummaryWriter('')
    global_train_step = 0
    global_val_step = 0
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        global_val_step = global_val_step + 1
        log.info(f'{LogColors.RED}Start epoch {epoch}{LogColors.RESET}')

        batch_id, batch_id_sp, loss_item, loss_value_seg_item = train(train_data_loader, 
                                                                      model, 
                                                                      optimizer, 
                                                                      scheduler, 
                                                                      epoch, 
                                                                      loss_seg,
                                                                      writer, 
                                                                      global_train_step,
                                                                      sets)
        
        avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
        log.info(
                f'{LogColors.YELLOW}Batch: {epoch}-{batch_id} ({batch_id_sp}), loss = {loss_item}, loss_seg = {loss_value_seg_item}, avg_batch_time = {avg_batch_time}{LogColors.RESET}')
        
        roc_auc, accuracy, sensitivity, specificity = validate(val_data_loader, 
                                                                 model,
                                                                 writer,
                                                                global_val_step,
                                                                 sets)
        
        if not sets.ci_test:
            if roc_auc > best_auc or epoch % 4 == 0:
                count += 1
                best_auc = roc_auc
                model_save_path = '{}_{}_epoch_{}_batch_{}_auc_{}.pth.tar'.format(save_folder, count, epoch, batch_id, roc_auc)
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                
                log.info(f'{LogColors.PURPLE}Save checkpoints: epoch = {epoch}, batch_id = {batch_id}{LogColors.RESET}') 
                torch.save({
                            'epoch': epoch,
                            'batch_id': batch_id,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            model_save_path)
    
    writer.close()
                            
    print('Finished training')            
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    sets = parse_opts()   
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt' 
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28

    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 

    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    elif sets.pretrain_path:
        
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate *100}
                ]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
        
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))
            
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    
      
    set_determinism(seed=12345)

    gs = GaussianSmooth()
    rgn = RandGaussianNoise()
    rz = RandZoom(prob=0.2)

    transforms = Compose([
        OneOf([gs, rgn]),
        rz 
    ])

    transforms = None
    training_dataset = ProstateDataset(sets.data_root, transforms, sets)
    train_indices, val_indices = train_test_split(range(len(training_dataset)), test_size=sets.validation_split, random_state=sets.manual_seed)
    train_dataset = torch.utils.data.Subset(training_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(training_dataset, val_indices)
    train_data_loader = DataLoader(train_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    val_data_loader = DataLoader(val_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    main(train_data_loader, val_data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_folder=sets.save_folder, sets=sets) 
