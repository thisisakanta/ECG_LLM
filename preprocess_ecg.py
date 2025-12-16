import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
import os
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import yaml
import sys
import json
# from utils_ecg import ECG_TEXT_Dsataset


# def preprocess_data():
#     torch.manual_seed(42)
#     random.seed(0)
#     np.random.seed(0)
#     # set up
#     config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

#     # loading data path
#     text_path = config['dataset']['text_path']
#     ecg_path = config['dataset']['ecg_path']

#     train_dataset = ECG_TEXT_Dsataset(
#         ecg_path=ecg_path, csv_path=text_path, dataset_name=config['dataset']['dataset_name'])
#     train_dataset = train_dataset.get_dataset(train_test='train')

#     print('Return the dataset successfully! ')


def build_instruct_dataset(ecg_name='mimic', save_dir=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Build instruction dataset with proper train/validation/test split
    
    Args:
        ecg_name: 'mimic' or 'ptbxl'
        save_dir: Directory to save train/val/test jsonl files
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
    """
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    
    # set up
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    # loading data path
    text_path = config['dataset']['text_path']
    ecg_path = config['dataset']['ecg_path']

    csv = pd.read_csv(text_path, low_memory=False)

    # Create save directory if it doesn't exist
    if save_dir is None:
        save_dir = '/home/user/Downloads/2005047/MEIT/ECG_LLMs/ptbxl'
    os.makedirs(save_dir, exist_ok=True)

    if ecg_name == 'ptbxl':
        text_csv = csv
        jsonl_data = []
        
        for idx in range(text_csv.shape[0]):
            current_line = {}

            # Use English translated report
            report = text_csv['report_en'].iloc[idx]
            # only keep not NaN
            if pd.isna(report):
                continue
            # Keep original case for English reports (they're already translated)
            report = str(report).strip()

            if 'something wrong with the data' in report.lower():
                pass
            else:
                current_line["dataset"] = ecg_name
                current_line["id"] = ecg_name + "_" + str(idx)
                current_line["ecg_path"] = os.path.join(ecg_path, text_csv['filename_lr'].iloc[idx]).replace('records100', 'records500').replace('_lr', '_hr')

                prompt = {}
                prompt["role"] = "user"
                prompt["content"] = "" # will add in the main code

                answer = {}
                answer["role"] = "assistant"
                answer["content"] = report # output of the proposed prompt

                current_line["messages"] = [prompt, answer]
                jsonl_data.append(current_line)

        # Shuffle data for random split
        random.shuffle(jsonl_data)
        
        # Calculate split indices
        total_samples = len(jsonl_data)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        # Split data
        train_data = jsonl_data[:train_end]
        val_data = jsonl_data[train_end:val_end]
        test_data = jsonl_data[val_end:]
        
        # Save train set
        train_path = os.path.join(save_dir, f'{ecg_name}_ecg_train.jsonl')
        with open(train_path, 'w') as file:
            for item in train_data:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f' Saved {len(train_data)} training samples to {train_path}')
        
        # Save validation set
        val_path = os.path.join(save_dir, f'{ecg_name}_ecg_val.jsonl')
        with open(val_path, 'w') as file:
            for item in val_data:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f' Saved {len(val_data)} validation samples to {val_path}')
        
        # Save test set
        test_path = os.path.join(save_dir, f'{ecg_name}_ecg_test.jsonl')
        with open(test_path, 'w') as file:
            for item in test_data:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f' Saved {len(test_data)} test samples to {test_path}')

    elif ecg_name == 'mimic':
        csv = csv.sort_values(by=['study_id']) # csv for the ecg text document
        csv.reset_index(inplace=True, drop=True)
        print(f'total csv size: {csv.shape[0]}')

        record_csv = pd.read_csv(os.path.join(ecg_path, 'record_list.csv'), low_memory=False)
        record_csv = record_csv.sort_values(by=['study_id']) # csv for the path_index of ecg signal
        record_csv.reset_index(inplace=True, drop=True)

        text_csv = csv
        jsonl_data = []
        
        for idx in range(record_csv.shape[0]):
            current_line = {}

            report = text_csv.iloc[idx][['report_0', 'report_1',
                'report_2', 'report_3', 'report_4', 'report_5', 'report_6', 'report_7',
                'report_8', 'report_9', 'report_10', 'report_11', 'report_12',
                'report_13', 'report_14', 'report_15', 'report_16', 'report_17']]
            # only keep not NaN
            report = report[~report.isna()]
             # concat the report
            report = '. '.join(report)
            # preprocessing on raw text
            report = report.replace('EKG', 'ECG')
            report = report.replace('ekg', 'ecg')
            report = report.strip('*** ')
            report = report.strip(' ***')
            report = report.strip('***')
            report = report.strip('=-')
            report = report.strip('=')

            # convert to all lower case
            report = report.lower()

            current_line["dataset"] = ecg_name
            current_line["id"] = ecg_name + "_" + str(idx)
            current_line["ecg_path"] = os.path.join(ecg_path, record_csv['path'].iloc[idx])

            prompt = {}
            prompt["role"] = "user"
            prompt["content"] = "" # will add in the main code

            answer = {}
            answer["role"] = "assistant"
            answer["content"] = report # output of the proposed prompt

            current_line["messages"] = [prompt, answer]
            jsonl_data.append(current_line)

        # Shuffle data for random split
        random.shuffle(jsonl_data)
        
        # Calculate split indices
        total_samples = len(jsonl_data)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        # Split data
        train_data = jsonl_data[:train_end]
        val_data = jsonl_data[train_end:val_end]
        test_data = jsonl_data[val_end:]
        
        # Save train set
        train_path = os.path.join(save_dir, f'{ecg_name}_ecg_train.jsonl')
        with open(train_path, 'w') as file:
            for item in train_data:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f' Saved {len(train_data)} training samples to {train_path}')
        
        # Save validation set
        val_path = os.path.join(save_dir, f'{ecg_name}_ecg_val.jsonl')
        with open(val_path, 'w') as file:
            for item in val_data:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f' Saved {len(val_data)} validation samples to {val_path}')
        
        # Save test set
        test_path = os.path.join(save_dir, f'{ecg_name}_ecg_test.jsonl')
        with open(test_path, 'w') as file:
            for item in test_data:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f' Saved {len(test_data)} test samples to {test_path}')
        
    print(f'\ Dataset split summary:')
    print(f'   Total samples: {total_samples}')
    print(f'   Train: {len(train_data)} ({train_ratio*100:.0f}%)')
    print(f'   Val: {len(val_data)} ({val_ratio*100:.0f}%)')
    print(f'   Test: {len(test_data)} ({test_ratio*100:.0f}%)')

if __name__ == '__main__':
    print("=" * 70)
    print("ECG DATA PREPROCESSING - Train/Val/Test Split")
    print("=" * 70)
    
    # PTB-XL dataset with proper train/val/test split
    print("\n📊 Processing PTB-XL dataset...")
    build_instruct_dataset(
        ecg_name='ptbxl',
        save_dir='/home/user/Downloads/2005047/MEIT/ECG_LLMs/ptbxl',
        train_ratio=0.8,  # 80% training
        val_ratio=0.1,    # 10% validation
        test_ratio=0.1    # 10% test
    )
    
    print("\n" + "=" * 70)
    print("✅ Dataset preprocessing completed!")
    print("=" * 70)
    print("\n📁 Generated files:")
    print("   - ptbxl_ecg_train.jsonl (80% of data)")
    print("   - ptbxl_ecg_val.jsonl (10% of data)")
    print("   - ptbxl_ecg_test.jsonl (10% of data - use for inference)")
    
    # Uncomment to process MIMIC-IV dataset
    # print("\n📊 Processing MIMIC-IV dataset...")
    # build_instruct_dataset(
    #     ecg_name='mimic',
    #     save_dir='E:/meit/data',
    #     train_ratio=0.8,
    #     val_ratio=0.1,
    #     test_ratio=0.1
    # )

