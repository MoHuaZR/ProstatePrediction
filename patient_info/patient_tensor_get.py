import json
import pandas as pd
import torch

class Clinic_Info(object):
    def __init__(self, excel_path, json_path, train_or_test):
        self.colums_atr = {'num':'编号', 'name':'姓名', 'age':'年龄', 'psa':'PSA',
                    'isup':'活检ISUP分组', 't_time':'临床T分期', 'suvmax':'SUVmax'}

        self.excel_path = excel_path
        self.json_path = json_path
        self.mix_patient_info = []
        self.train_or_test = train_or_test
        
    def read_excel(self, excel_path):
        df = pd.read_excel(excel_path)
        return df
    def read_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    def judge_type(self, item):
        if isinstance(item, str):
            return eval(item.strip())
        return item
    
    def process_info(self, patient_list):
        for name in patient_list: 
            name_to_find = self.read_json(self.json_path)[self.train_or_test][name]
            df = self.read_excel(self.excel_path)
            result = df.loc[df[self.colums_atr['name']] == name_to_find]    
            NUM = result[self.colums_atr['num']].values[0]       
            AGE = result[self.colums_atr['age']].values[0]  
            PSA = result[self.colums_atr['psa']].values[0]  
            ISUP = result[self.colums_atr['isup']].values[0]  
            TTIMES = result[self.colums_atr['t_time']].values[0]
            SUV_MAX = result[self.colums_atr['suvmax']].values[0] 

            single_patient = list((self.judge_type(AGE), 
                                   self.judge_type(PSA), 
                                   self.judge_type(ISUP), 
                                   self.judge_type(TTIMES), 
                                   self.judge_type(SUV_MAX),
                                   self.judge_type(NUM)))
            self.mix_patient_info.append(single_patient)
            X_tensor = torch.Tensor(self.mix_patient_info)
            
        return X_tensor



