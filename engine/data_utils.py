#!/usr/bin/env python
# coding=utf-8

import os
import pickle
import logging
import tqdm
import json
import random

import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

from engine.utils import make_LHE


class MyDataset(Dataset):
    def __init__(self, full_data, need_label=True):
        self.data = full_data
        random.shuffle(self.data)
        self.need_label = need_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        masks_list = self.data[idx]['masks']
        input_ids = self.data[idx]['input_ids']
        if self.need_label:
            label_ids = self.data[idx]['output_ids']
            data_dict = {
                'input_ids': input_ids,
                'masks': masks_list,
                'output_ids': label_ids,
            }
        else:
            data_dict = {
                'input_ids': input_ids,
                'masks': masks_list,
            }

        return data_dict

    def collate_fn(self, batch):
        input_ids_list = []
        masks_list = []
        output_ids_list = []
        bsz = 0
        for data in batch:
            bsz += 1
            input_ids_list.append(data['input_ids'])
            masks_list.append(data['masks'])
            if self.need_label:
                output_ids_list.append(data['output_ids'])
        input_ids = torch.stack(input_ids_list)
        masks = torch.stack(masks_list)
        if self.need_label:
            output_ids = torch.stack(output_ids_list)
            return {
                'input_ids': input_ids,
                'masks': masks,
                'output_ids': output_ids,
            }
        else:
            return {
                'input_ids': input_ids,
                'masks': masks,
            }



def create_data(tokenizer, file_, task, task_type, span_attribute_list, rel_type_list, max_input_seq_length, max_output_seq_length,
                need_target_LHE=True):
    data_ = []

    with open(file_, 'r', encoding='utf8') as f:
        json_file = json.load(f)

    for i in tqdm.tqdm(range(len(json_file))):
        sentence = json_file[i]['sentence']

        if need_target_LHE:
            task_type_content = json_file[i][task_type + 's']
        else:
            task_type_content = None

        input_seq, output_seq = make_LHE(task_type_content, task_type, task, sentence, span_attribute_list, rel_type_list, need_target_LHE)

        tokened_input = tokenizer(input_seq, max_length=max_input_seq_length, truncation=True, padding='max_length', return_tensors="pt",
                                  return_attention_mask=True)
        input_id = tokened_input.input_ids.squeeze()
        attn_mask = tokened_input.attention_mask.squeeze()

        if output_seq == None:
            temp_dict = {'index': i,
                         'input_ids': input_id,
                         'masks': attn_mask,
                         }
        else:
            output_id = tokenizer(output_seq, max_length=max_output_seq_length, truncation=True, padding='max_length', return_tensors="pt")[
                'input_ids'].squeeze()

            temp_dict = {'index': i,
                         'input_ids': input_id,
                         'masks': attn_mask,
                         'output_ids': output_id,
                         }

        data_.append(temp_dict)

    return data_


def save_pkl(data, pkl_path):
    data_file = open(pkl_path, 'ab+')
    for i in range(len(data)):
        pickle.dump(data[i], data_file)


def get_labels(label_file, task):
    if os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        if task in labels.keys():
            task_name = labels[task]
        else:
            raise RuntimeError("no task label defined")

        if 'span-attribute' in labels.keys():
            span_attribute_list = []
            for k, v in labels['span-attribute'].items():
                span_attribute_list.append(v)
            span_attribute_list = list(set(span_attribute_list))
        else:
            raise RuntimeError("no span labels defined")

        if 'relation-type' in labels.keys():
            rel_type_list = []
            for k, v in labels['relation-type'].items():
                rel_type_list.append(v)
            rel_type_list = list(set(rel_type_list))

        else:
            rel_type_list = None

        return span_attribute_list, rel_type_list, task_name

    else:
        raise FileNotFoundError


def loading_data(tokenizer_location, file_path, task, task_type, train_file, dev_file, test_file, label_file,
                 max_input_seq_length, max_output_seq_length, logger=None, is_save=False, need_target_LHE=True):
    if logger == None:
        logger = logging.getLogger(__name__)

    logger.info('*' * 20 + 'loading data in ' + file_path + ' data' + '*' * 20)

    label_file_path = os.path.join(file_path, label_file)
    span_attribute_list, rel_type_list, task_name = get_labels(label_file_path, task)

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_location)

    if train_file != None:
        train_file_prefix = train_file.split('.')[0]

        # train set
        pkl_train_file = os.path.join(file_path, train_file_prefix + '.pkl')
        if os.path.exists(pkl_train_file):
            logger.info('*' * 10 + 'load from existing ' + train_file_prefix + ' data' + '*' * 10)
            train_data_file = open(pkl_train_file, 'rb')
            train_data = []
            while True:
                try:
                    temp_data = pickle.load(train_data_file)
                    train_data.append(temp_data)
                except EOFError:
                    break
            train_data_file.close()
        else:
            logger.info('*' * 10 + 'make ' + train_file_prefix + ' data' + '*' * 10)
            json_train_file = os.path.join(file_path, train_file_prefix + '.json')
            train_data = create_data(tokenizer, json_train_file, task_name, task_type, span_attribute_list, rel_type_list, max_input_seq_length,
                                     max_output_seq_length)
            if is_save:
                save_pkl(train_data, pkl_train_file)

        logger.info(f'train set number: {len(train_data)}')
    else:
        train_data = None

    if dev_file != None:
        dev_file_prefix = dev_file.split('.')[0]
        # dev set
        pkl_dev_file = os.path.join(file_path, dev_file_prefix + '.pkl')
        if os.path.exists(pkl_dev_file):
            logger.info('*' * 10 + 'load from existing ' + dev_file_prefix + ' data' + '*' * 10)
            dev_data_file = open(pkl_dev_file, 'rb')
            dev_data = []
            while True:
                try:
                    temp_data = pickle.load(dev_data_file)
                    dev_data.append(temp_data)
                except EOFError:
                    break
            dev_data_file.close()
        else:
            logger.info('*' * 10 + 'make ' + dev_file_prefix + ' data' + '*' * 10)
            json_dev_file = os.path.join(file_path, dev_file_prefix + '.json')
            dev_data = create_data(tokenizer, json_dev_file, task_name, task_type, span_attribute_list, rel_type_list, max_input_seq_length,
                                   max_output_seq_length)
            if is_save:
                save_pkl(dev_data, pkl_dev_file)

        logger.info(f'dev set number: {len(dev_data)}')

    else:
        dev_data = None

    if test_file != None:
        test_file_prefix = test_file.split('.')[0]

        # test set
        pkl_test_file = os.path.join(file_path, test_file_prefix + '.pkl')
        if os.path.exists(pkl_test_file):
            logger.info('*' * 10 + 'load from existing ' + test_file_prefix + ' data' + '*' * 10)
            test_data_file = open(pkl_test_file, 'rb')
            test_data = []
            while True:
                try:
                    temp_data = pickle.load(test_data_file)
                    test_data.append(temp_data)
                except EOFError:
                    break
            test_data_file.close()
        else:
            logger.info('*' * 10 + 'make ' + test_file_prefix + ' data' + '*' * 10)
            json_test_file = os.path.join(file_path, test_file_prefix + '.json')
            test_data = create_data(tokenizer, json_test_file, task_name, task_type, span_attribute_list, rel_type_list, max_input_seq_length,
                                    max_output_seq_length, need_target_LHE=need_target_LHE)
            if is_save:
                save_pkl(test_data, pkl_test_file)

        logger.info(f'test set number: {len(test_data)}')

    else:
        test_data = None

    return train_data, dev_data, test_data
