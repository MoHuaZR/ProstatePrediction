from torch.utils.data import Dataset
import os
import nibabel
import numpy as np
from scipy import ndimage
import torch

class ProstateDataset(Dataset):
    def __init__(self, root_dir, transform, sets):
        self.root_dir = root_dir   
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.image_filenames_ct = []
        self.labels_train = []
        self.image_filenames_pet = []
        self.labels_test = []
        self.phase = sets.phase
        self.is_transform = sets.if_transform
        self.transform = transform
        if self.phase == "train":
            classpath = os.path.join(self.root_dir, 'train')
        else:
            classpath = os.path.join(self.root_dir, 'test')
        classtype = os.listdir(classpath)
        for cls_id, cls_name in enumerate(classtype):
            class_path_ct = os.path.join(''.join([classpath, '/',cls_name]), 'CT')    
            class_path_pet = os.path.join(''.join([classpath, '/',cls_name]), 'PET')     
            for filename in os.listdir(class_path_ct):
                self.image_filenames_ct.append(os.path.join(class_path_ct, filename))
                self.image_filenames_pet.append(os.path.join(class_path_pet, filename))
                self.labels_train.append(cls_id)
                if self.phase == 'test':
                    self.labels_test.append(cls_id)
            
    def __len__(self):
        return len(self.image_filenames_ct)
    
    def __check_data__(self, patient_name, img_path_ct, img_path_pet, img_array_ct, img_array_pet, phase):
        print("-----------------------CHECKED----------------------")
        if phase == 'train':
            parent_path = '...'
        else:
            parent_path = '...'
        file_name = ''.join([patient_name, '.nii.gz'])
        ct = img_path_ct.split('/')[-2]
        pet = img_path_pet.split('/')[-2]
        neg_or_posi = img_path_pet.split('/')[-3]
        print("ct_or_pet:", ct)
        path_ct = os.path.join(parent_path, neg_or_posi, ct, file_name)
        path_pet = os.path.join(parent_path, neg_or_posi, pet, file_name)
        nifti_img_ct = nibabel.Nifti1Image(img_array_ct, affine=np.eye(4))
        nifti_img_pet = nibabel.Nifti1Image(img_array_pet, affine=np.eye(4))
        nibabel.save(nifti_img_ct, path_ct)
        nibabel.save(nifti_img_pet, path_pet)

    def __getitem__(self, idx):
        img_path_ct = self.image_filenames_ct[idx]  
        img_path_pet = self.image_filenames_pet[idx]
        assert os.path.isfile(img_path_ct)
        assert os.path.isfile(img_path_pet)
        patient_name =  img_path_ct.split('/')[-1].split('.')[0]
        if self.phase == "train":
            img_array_ct = self.__training_data_process__(img_path_ct, patient_name, is_ct = True)
            img_array_pet = self.__training_data_process__(img_path_pet, patient_name, is_ct = False)  
            img_array_ct = self.__nii2tensorarray__(img_array_ct)
            img_array_pet = self.__nii2tensorarray__(img_array_pet)                  
            if self.is_transform:  
                img_array_ct = self.transform(img_array_ct)
                img_array_pet = self.transform(img_array_pet)
            label = torch.tensor(self.labels_train[idx], dtype = torch.long)
            phase = 'train' 
        if self.phase =='test':
            img_array_ct = self.__testing_data_process__(img_path_ct, is_ct = True)
            img_array_pet = self.__testing_data_process__(img_path_pet, is_ct = False)  
            img_array_ct = self.__nii2tensorarray__(img_array_ct)
            img_array_pet = self.__nii2tensorarray__(img_array_pet)
            label = torch.tensor(self.labels_train[idx], dtype = torch.long)
            phase = 'test'
        return img_array_ct, img_array_pet, label, patient_name, phase  

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out
    
    def __resize_data__(self, data):
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)
        return data

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
        return new_data
        
    def __read_Nifit__(self, path):
        nii_img = nibabel.load(path)
        nii_data = nii_img.get_fdata()
        return nii_data
    
    def __get_shape__(self, patient_name , path):
        nifti_img_raw = nibabel.load(path)
        data_raw = nifti_img_raw.get_fdata()
        dimensions_raw = data_raw.shape
        print("{} shape: {}".format(patient_name, dimensions_raw))      

    def __training_data_process__(self, data, patient_name, is_ct): 
        assert data is not None
        data = self.__read_Nifit__(data)
        data = self.__resize_data__(data)
        if is_ct:
            data = self.__itensity_normalize_one_volume__(data)
        else:
            data = self.__itensity_normalize_one_volume__(data)  
        return data
        
    def __testing_data_process__(self, data, is_ct): 
        assert data is not None
        data = self.__read_Nifit__(data)
        data = self.__resize_data__(data)
        if is_ct:
            data = self.__itensity_normalize_one_volume__(data)
        else:
            data = self.__itensity_normalize_one_volume__(data)
        return data
    
