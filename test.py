from setting import parse_opts 
from datasets.prostate import ProstateDataset 
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from patient_info.patient_tensor_get import Clinic_Info
from utils.logger import log
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize
import json

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

def test(test_data_loader, model, save_path, sets):
    model.eval()
    predicts = []
    gts = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_data_loader):       
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
                outputs = model(volumes_ct, volumes_pet, tensor_clinic)
            else:
                outputs = model(volumes_ct, volumes_pet, tensor_clinic = None)
            outputs_logit = outputs.argmax(dim=1) 
            predicts.append(outputs_logit.cpu().detach().numpy())
            gts.append(class_array.cpu().detach().numpy())
        
        predicts = np.concatenate(predicts).flatten().astype(np.int16)
        gts = np.concatenate(gts).flatten().astype(np.int16)
        
        fpr, tpr, _ = roc_curve(gts, predicts)
        roc_auc = auc(fpr, tpr)
        excel_file_path = ''
        excel_data = pd.read_excel(excel_file_path)
        folder_path = ''
        file_list = os.listdir(folder_path)
        psma_values = []
        mskcc_values = []
        labels = []
        with open(sets.json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for file_name in file_list:
            patient_id = os.path.splitext(file_name)[0]
            name = patient_id.split('.')[0]
            name_to_find = data['test'][name]
            patient_data = excel_data[excel_data['姓名'] == name_to_find]
            if not patient_data.empty:
                psma_values.append(patient_data['PSMA'].values[0])
                mskcc_values.append(patient_data['MSKCC'].values[0])
                labels.append(patient_data['淋巴结转移情况'].values[0])
        psma_values = np.array(psma_values)
        mskcc_values = np.array(mskcc_values)
        labels = np.array(labels)
        labels = label_binarize(labels, classes=[0, 1])
        fpr_psma, tpr_psma, _ = roc_curve(labels, psma_values)
        roc_auc_psma = auc(fpr_psma, tpr_psma)

        fpr_mskcc, tpr_mskcc, _ = roc_curve(labels, mskcc_values)
        roc_auc_mskcc = auc(fpr_mskcc, tpr_mskcc)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'OURS (AUC = {roc_auc:.4f})')
        plt.plot(fpr_psma, tpr_psma, color='blue', lw=1.5, label='PSMA (AUC = {:.2f})'.format(roc_auc_psma))
        plt.plot(fpr_mskcc, tpr_mskcc, color='red', lw=1.5, label='MSKCC (AUC = {:.2f})'.format(roc_auc_mskcc))
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Charactericstic Curve (ROC)')
        plt.legend(loc='lower right')
        name_roc = 'roc.jpg'
        name_confusion = 'confusion_matrix.jpg'
        save_path_roc = ''.join([save_path, '/', name_roc]) 
        save_path_matrix = ''.join([save_path, '/', name_confusion]) 
        if save_path_roc:
            plt.savefig(save_path_roc)
        else:
            plt.show()
        cm = confusion_matrix(gts, predicts)
        classes = unique_labels(gts, predicts)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        plt.figure()
        disp.plot(cmap=plt.cm.Blues, values_format='.4g')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        if save_path_matrix:
            plt.savefig(save_path_matrix)
        else:
            plt.show()
        roc_auc, accuracy, sensitivity, specificity = caculate_metrics(gts, predicts)
    return roc_auc, accuracy, sensitivity, specificity


if __name__ == '__main__':
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])
    transform = None
    testing_data = ProstateDataset(sets.data_root, transform, sets)
    data_loader = DataLoader(testing_data, batch_size=50, shuffle=False, num_workers=8, pin_memory=False)
    save_path = ''
    roc_auc, accuracy, sensitivity, specificity = test(data_loader, net, save_path, sets)
    
